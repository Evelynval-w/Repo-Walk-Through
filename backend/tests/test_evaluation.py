import json
from pathlib import Path

import pytest

from app.evaluation.judge import JudgeError, judge_facts, judge_refusal
from app.evaluation.questions import Question, load_questions
from app.evaluation.runner import file_of, format_report, run_all, summarize
from app.ingest.embedder import FakeEmbedder
from app.ingest.models import Chunk
from app.ingest.store import ChunkStore
from app.qa.chat import ChatError, FakeChat

REAL_FILE = Path(__file__).resolve().parents[2] / "eval" / "questions.toml"


class ScriptedChat:
    """Returns the replies in order. Used for the judge."""
    name = "scripted"

    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []

    def generate_json(self, system, user):
        self.calls.append(user)
        from app.qa.chat import ChatReply
        return ChatReply(self.texts.pop(0), self.name)


@pytest.fixture
def store(tmp_path):
    fake = FakeEmbedder()
    store = ChunkStore(tmp_path / "db", fake.name)
    chunks = [
        Chunk("airline", "Booking.java", "Booking.book", "method", "java", "book a flight and check seats", 1, 9),
        Chunk("airline", "Miles.java", "Miles.reward", "method", "java", "discount code every third flight", 1, 9),
    ]
    store.upsert(chunks, fake.embed_documents([c.embed_text() for c in chunks]))
    return store


ANSWERABLE = Question("ta-x", "airline", "how", "book a flight and check seats", ["airline/Booking.java"], ["checks seats", "books"])
UNANSWERABLE = Question("un-x", "airline", "unanswerable", "how are payments taken", [], [], False)


def answer_chat(used=(1,)):
    return FakeChat(json.dumps({"answer": "Booking.book checks seats and books.", "used": list(used)}))


def go(questions, store, chat, judge, **kw):
    return run_all(questions, embedder=FakeEmbedder(), store=store, chat=chat, judge_chat=judge, top_k=2, **kw)


def test_file_of_strips_the_chunk_part():
    assert file_of("repo/a/b.py#function:f@3-9:0") == "repo/a/b.py"


def test_a_good_run_scores_all_three_numbers(store):
    judge = ScriptedChat('{"facts": [true, true]}')
    [r] = go([ANSWERABLE], store, answer_chat(), judge)
    assert (r.retrieval_hit, r.citation_hit, r.correct) == (True, True, True)


def test_found_but_not_cited_is_a_retrieval_hit_and_a_citation_miss(store):
    # top_k=2 shows both chunks; the model cites only Miles.java (excerpt number depends on ranking)
    chat = FakeChat(json.dumps({"answer": "x", "used": []}))
    [r] = go([ANSWERABLE], store, chat, ScriptedChat('{"facts": [false, false]}'))
    assert r.retrieval_hit is True and r.citation_hit is False and r.correct is False


def test_one_missing_fact_makes_the_answer_wrong(store):
    [r] = go([ANSWERABLE], store, answer_chat(), ScriptedChat('{"facts": [true, false]}'))
    assert r.correct is False and r.facts_found == [True, False]


def test_a_failed_answer_counts_as_a_miss_not_a_skip(store):
    class Down:
        name = "down"
        def generate_json(self, s, u):
            raise ChatError("Gemini returned 429", retryable=True)
    [r] = go([ANSWERABLE], store, Down(), None)
    assert r.error and r.correct is False and r.retrieval_hit is False


def test_bad_judge_output_counts_as_wrong_and_is_recorded(store):
    [r] = go([ANSWERABLE], store, answer_chat(), ScriptedChat("not json"))
    assert r.correct is False and "judge" in r.judge_error


def test_unanswerable_questions_are_not_scored_but_refusal_is_judged(store):
    [r] = go([UNANSWERABLE], store, answer_chat(), ScriptedChat('{"refused": false}'))
    assert r.correct is None and r.retrieval_hit is None and r.refused is False
    summary = summarize([UNANSWERABLE], [r])
    assert summary["groups"] == {} and summary["unanswerable"]["asked"] == 1


def test_runs_repeat_each_question_and_pause_between_calls(store):
    pauses = []
    results = go([ANSWERABLE], store, answer_chat(), None, runs=3, pause=1.5, sleep=pauses.append)
    assert [r.run for r in results] == [1, 2, 3] and pauses == [1.5, 1.5, 1.5]


def test_summary_groups_by_repo_and_kind_and_the_report_lists_misses(store):
    judge = ScriptedChat('{"facts": [true, true]}', '{"facts": [true, false]}')
    results = go([ANSWERABLE, Question("ta-y", "airline", "list", "list the booking steps", ["airline/Booking.java"], ["a", "b"])], store, answer_chat(), judge)
    qs = [ANSWERABLE, Question("ta-y", "airline", "list", "list the booking steps", ["airline/Booking.java"], ["a", "b"])]
    summary = summarize(qs, results)
    assert summary["groups"]["all"]["correct"] == {"hit": 1, "total": 2}
    assert summary["groups"]["kind:list"]["correct"] == {"hit": 0, "total": 1}
    report = format_report(qs, results, summary, {"top_k": 2})
    assert "ta-y run 1" in report and "Hand check" in report


def test_judge_checks_the_number_of_flags():
    with pytest.raises(JudgeError):
        judge_facts(ScriptedChat('{"facts": [true]}'), "q", "a", ["one", "two"])
    with pytest.raises(JudgeError):
        judge_refusal(ScriptedChat('{"refused": "yes"}'), "q", "a")
    assert judge_facts(ScriptedChat('```json\n{"facts": [true, false]}\n```'), "q", "a", ["one", "two"]) == [True, False]


def test_loader_rejects_bad_entries(tmp_path):
    p = tmp_path / "q.toml"
    p.write_text('[[questions]]\nid="a"\nrepo="r"\nkind="k"\nquestion="q"\nanswerable=true\n')
    with pytest.raises(ValueError, match="needs expected_files"):
        load_questions(p)


def test_the_real_question_file_is_valid_and_every_expected_file_exists():
    questions = load_questions(REAL_FILE)
    assert len(questions) == 37 and sum(bool(q.evidence) for q in questions) == 31
    assert sum(q.answerable for q in questions) == 31
    repos = Path(__file__).resolve().parents[2] / "data" / "repos"
    if not repos.exists():
        pytest.skip("repos are not cloned here")
    missing = [(q.id, f) for q in questions for f in q.expected_files if not (repos / f).exists()]
    assert missing == []


def test_summary_counts_facts_separately_from_all_or_nothing(store):
    judge = ScriptedChat('{"facts": [true, false]}')
    results = go([ANSWERABLE], store, answer_chat(), judge)
    s = summarize([ANSWERABLE], results)["groups"]["all"]
    assert s["correct"] == {"hit": 0, "total": 1} and s["facts"] == {"hit": 1, "total": 2}


def test_show_names_where_a_run_went_wrong(store):
    from app.evaluation.runner import results_to_json
    from app.evaluation.show import diagnose, render
    results = go([ANSWERABLE], store, answer_chat(), ScriptedChat('{"facts": [true, false]}'))
    report = {"results": results_to_json(results)}
    assert diagnose(ANSWERABLE, report["results"][0]).startswith("ANSWER")
    text = render([ANSWERABLE], report)
    assert "[x] checks seats" in text and "[ ] books" in text
    miss = dict(report["results"][0], retrieval_hit=False, citation_hit=False)
    assert diagnose(ANSWERABLE, miss).startswith("SEARCH")


def test_rejudge_scores_saved_answers_again_without_asking_the_question_again(store):
    from app.evaluation.rejudge import rejudge
    from app.evaluation.runner import results_to_json
    first = go([ANSWERABLE], store, answer_chat(), ScriptedChat('{"facts": [true, false]}'))
    assert first[0].correct is False
    fixed = Question("ta-x", "airline", "how", "book a flight and check seats", ["airline/Booking.java"], ["checks seats"])
    judge = ScriptedChat('{"facts": [true]}')
    [again] = rejudge([fixed], results_to_json(first), judge)
    assert again.correct is True and again.facts_found == [True]
    assert again.answer == first[0].answer and again.retrieval_hit is True  # search result is kept
    assert len(judge.calls) == 1


def test_rejudge_keeps_failed_runs_as_they_were(store):
    from app.evaluation.rejudge import rejudge
    failed = {"question_id": "ta-x", "run": 1, "error": "ChatError: down", "retrieval_hit": False, "citation_hit": False, "correct": False}
    [r] = rejudge([ANSWERABLE], [failed], ScriptedChat())
    assert r.error and r.correct is False


def test_evidence_is_checked_against_the_text_of_the_shown_chunks(store):
    q = Question("ta-e", "airline", "how", "book a flight and check seats", ["airline/Booking.java"],
                 ["checks seats"], True, ["book a FLIGHT   and check seats", "discount code every third flight"])
    [r] = go([q], store, answer_chat(), ScriptedChat('{"facts": [true]}'))
    assert r.evidence_found == [True, True] and r.evidence_hit is True  # case and spacing are ignored

    missing = Question("ta-f", "airline", "how", "book a flight and check seats", ["airline/Booking.java"],
                       ["checks seats"], True, ["refund policy"])
    [r2] = go([missing], store, answer_chat(), ScriptedChat('{"facts": [true]}'))
    assert r2.retrieval_hit is True and r2.evidence_hit is False  # right file, answer chunk not shown


def test_questions_without_evidence_are_not_counted_in_the_evidence_column(store):
    results = go([ANSWERABLE], store, answer_chat(), ScriptedChat('{"facts": [true, true]}'))
    assert results[0].evidence_hit is None
    assert summarize([ANSWERABLE], results)["groups"]["all"]["evidence_hit"] == {"hit": 0, "total": 0}


def test_a_failing_embedding_service_is_recorded_not_fatal(store):
    from app.ingest.embedder import EmbeddingError

    class BrokenEmbedder(FakeEmbedder):
        def embed_query(self, text):
            raise EmbeddingError("Gemini returned 503")
    results = run_all([ANSWERABLE], embedder=BrokenEmbedder(), store=store, chat=answer_chat(), judge_chat=None, top_k=2)
    assert "EmbeddingError" in results[0].error and results[0].correct is False


def test_show_reports_right_file_wrong_chunk(store):
    from app.evaluation.runner import results_to_json
    from app.evaluation.show import diagnose
    missing = Question("ta-f", "airline", "how", "book a flight and check seats", ["airline/Booking.java"],
                       ["checks seats"], True, ["refund policy"])
    [r] = go([missing], store, answer_chat(), ScriptedChat('{"facts": [false]}'))
    assert "wrong chunk" in diagnose(missing, results_to_json([r])[0]) or "not shown" in diagnose(missing, results_to_json([r])[0])


def test_every_evidence_phrase_exists_in_a_real_chunk_of_the_same_repo():
    """The phrases must come from the real code, never from memory."""
    from app.config import load_settings
    from app.ingest.pipeline import chunk_repo
    from app.ingest.walker import load_repo_configs
    settings = load_settings()
    if not settings.repos_dir.exists():
        pytest.skip("repos are not cloned here")
    texts = {}
    for cfg in load_repo_configs(settings.repos_config):
        root = settings.repos_dir / cfg.name
        if root.exists():
            texts[cfg.name] = [" ".join(c.text.lower().split()) for c in chunk_repo(cfg, root, settings).chunks]
    missing = []
    for q in load_questions(REAL_FILE):
        for phrase in q.evidence:
            if q.repo not in texts:
                continue
            if not any(" ".join(phrase.lower().split()) in t for t in texts[q.repo]):
                missing.append((q.id, phrase))
    assert missing == []
