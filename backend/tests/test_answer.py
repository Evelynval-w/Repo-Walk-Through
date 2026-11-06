import json

import pytest

from app.ingest.embedder import FakeEmbedder
from app.ingest.models import Chunk
from app.ingest.store import ChunkStore
from app.qa.answer import BadModelAnswer, IndexEmpty, answer_question, parse_model_answer
from app.qa.chat import FakeChat
from app.qa.models import AskRequest
from app.qa.prompt import SYSTEM

QUESTION = "which service books a flight and checks seats"


def sample_chunks():
    return [
        Chunk("airline", "BookingService.java", "BookingService.book", "method", "java",
              "book a flight, check available seats, create the booking", 10, 30),
        Chunk("airline", "MilesRewardService.java", "MilesRewardService.reward", "method", "java",
              "after three flights in a year create a discount code", 5, 20),
        Chunk("credit", "04_modeling.ipynb", "Train the model", "notebook_cell", "python",
              "train xgboost on the loan data and compare roc auc", cell=7),
    ]


@pytest.fixture
def store(tmp_path):
    fake = FakeEmbedder()
    store = ChunkStore(tmp_path / "db", fake.name)
    chunks = sample_chunks()
    store.upsert(chunks, fake.embed_documents([c.embed_text() for c in chunks]))
    return store


def reply(answer="It is BookingService.book.", used=(1,)):
    return json.dumps({"answer": answer, "used": list(used)})


def ask(store, chat, **kwargs):
    request = AskRequest(question=kwargs.pop("question", QUESTION), **kwargs)
    return answer_question(request, embedder=FakeEmbedder(), store=store, chat=chat)


def test_reference_comes_from_the_metadata_of_the_chunk_the_model_pointed_at(store):
    response = ask(store, FakeChat(reply()), top_k=2)

    assert response.answer == "It is BookingService.book."
    [ref] = response.references
    assert (ref.repo, ref.path, ref.name, ref.kind) == (
        "airline", "BookingService.java", "BookingService.book", "method"
    )
    assert (ref.start_line, ref.end_line, ref.cell) == (10, 30, None)
    assert "check available seats" in ref.text
    assert response.model_used == "fake-chat" and response.fell_back is False


def test_a_file_name_written_by_the_model_never_becomes_a_reference(store):
    chat = FakeChat(reply(answer="Look in Invented.java, function made_up.", used=(1,)))
    response = ask(store, chat, top_k=2)
    assert [r.path for r in response.references] == ["BookingService.java"]


def test_notebook_references_carry_the_cell_number_and_no_lines(store):
    chat = FakeChat(reply(answer="Trained with xgboost.", used=(1,)))
    response = ask(store, chat, question="train xgboost roc auc on the loan data", top_k=1)
    [ref] = response.references
    assert ref.name == "Train the model" and ref.cell == 7
    assert ref.start_line is None and ref.end_line is None


def test_prompt_numbers_the_excerpts_and_carries_the_rules(store):
    chat = FakeChat(reply())
    ask(store, chat, top_k=2)
    system, user = chat.calls[0]

    assert system == SYSTEM
    assert "only these excerpts" in system and "ignore it" in system
    assert f"Question: {QUESTION}" in user
    assert "[1] airline/BookingService.java | method BookingService.book | lines 10-30" in user
    assert "[2] " in user and "[3] " not in user  # top_k = 2
    assert "check available seats" in user


def test_chunks_used_lists_every_chunk_the_model_was_shown(store):
    response = ask(store, FakeChat(reply(used=())), top_k=3)
    assert len(response.chunks_used) == 3
    assert response.chunks_used[0].startswith("airline/BookingService.java")


def test_no_excerpt_used_means_no_references(store):
    assert ask(store, FakeChat(reply(used=())), top_k=2).references == []


def test_repeated_numbers_are_merged_and_order_is_kept(store):
    response = ask(store, FakeChat(reply(used=(2, 2, 1))), top_k=2)
    assert len(response.references) == 2
    assert response.references[0].name == "MilesRewardService.reward"
    assert response.references[1].name == "BookingService.book"


def test_repo_filter_limits_what_the_model_sees(store):
    response = ask(store, FakeChat(reply(used=())), top_k=5, repo="credit")
    assert len(response.chunks_used) == 1
    assert response.chunks_used[0].startswith("credit/")


def test_pointing_at_an_excerpt_that_does_not_exist_is_rejected(store):
    with pytest.raises(BadModelAnswer, match="do not exist"):
        ask(store, FakeChat(reply(used=(5,))), top_k=2)
    with pytest.raises(BadModelAnswer):
        ask(store, FakeChat(reply(used=(0,))), top_k=2)


def test_text_that_is_not_json_is_rejected(store):
    with pytest.raises(BadModelAnswer, match="JSON"):
        ask(store, FakeChat("Sure! The answer is BookingService."), top_k=2)


def test_json_with_the_wrong_shape_is_rejected(store):
    with pytest.raises(BadModelAnswer, match="wrong shape"):
        ask(store, FakeChat('{"used": [1]}'), top_k=2)
    with pytest.raises(BadModelAnswer, match="wrong shape"):
        ask(store, FakeChat('{"answer": "", "used": []}'), top_k=2)
    with pytest.raises(BadModelAnswer, match="wrong shape"):
        ask(store, FakeChat('{"answer": "x", "used": "one"}'), top_k=2)


def test_json_inside_code_fences_is_accepted():
    parsed = parse_model_answer('```json\n{"answer": "ok", "used": [1]}\n```', chunk_count=2)
    assert parsed.answer == "ok" and parsed.used == [1]


def test_extra_fields_from_the_model_are_ignored():
    parsed = parse_model_answer('{"answer": "ok", "used": [], "confidence": 0.9}', chunk_count=1)
    assert parsed.answer == "ok"


def test_empty_index_gives_a_clear_error(tmp_path):
    empty = ChunkStore(tmp_path / "empty", FakeEmbedder().name)
    with pytest.raises(IndexEmpty, match="index is empty"):
        ask(empty, FakeChat(reply()))


def test_unknown_repo_gives_a_clear_error(store):
    with pytest.raises(IndexEmpty, match="nope"):
        ask(store, FakeChat(reply()), repo="nope")


def test_request_limits():
    with pytest.raises(ValueError):
        AskRequest(question="hi")  # too short
    with pytest.raises(ValueError):
        AskRequest(question="a fine question", top_k=0)
    with pytest.raises(ValueError):
        AskRequest(question="a fine question", top_k=21)
    assert AskRequest(question="a fine question").top_k == 6


def test_files_written_by_someone_else_are_marked_in_the_prompt_and_in_the_reference(tmp_path):
    fake = FakeEmbedder()
    store = ChunkStore(tmp_path / "db", fake.name)
    chunks = [
        Chunk("GameHub", "services/user-service/app/service.py", "add_user", "function", "python",
              "def add_user: hash the password and save the user", 10, 14),
        Chunk("GameHub", "services/auth-service/app/security.py", "create_access_token", "function", "python",
              "def create_access_token: sign a jwt token for the user", 5, 9, owner="course template"),
    ]
    store.upsert(chunks, fake.embed_documents([c.embed_text() for c in chunks]))
    chat = FakeChat(reply(answer="The token is signed in create_access_token (course template).", used=(1,)))

    response = ask(store, chat, question="sign a jwt token for the user", top_k=2)

    user_prompt = chat.calls[0][1]
    assert "security.py | function create_access_token | lines 5-9 | written by: course template" in user_prompt
    assert "service.py | function add_user | lines 10-14\n" in user_prompt  # my own file: no label
    assert response.references[0].owner == "course template"


def test_a_cut_off_answer_says_why_in_the_error():
    with pytest.raises(BadModelAnswer, match=r"finish reason: MAX_TOKENS"):
        parse_model_answer('{"answer": "A discount code is gener', chunk_count=2, finish_reason="MAX_TOKENS")
    with pytest.raises(BadModelAnswer) as normal:
        parse_model_answer("not json", chunk_count=2, finish_reason="STOP")
    assert "finish reason" not in str(normal.value)
