import httpx

from app.tools import list_openrouter_models as tool

MODELS = [
    {"id": "paid/model", "pricing": {"prompt": "0.001", "completion": "0.002"}, "supported_parameters": ["response_format"]},
    {"id": "free/plain:free", "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["temperature"]},
    {"id": "free/json:free", "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["response_format"], "context_length": 8000},
    {"id": "free/structured:free", "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["structured_outputs"]},
    {"id": "no-pricing/model"},
]


def test_free_models_only_by_default_with_json_capable_first():
    ids = [m["id"] for m in tool.pick(MODELS, include_paid=False)]
    assert ids == ["free/json:free", "free/structured:free", "free/plain:free"]


def test_all_includes_paid_models():
    ids = [m["id"] for m in tool.pick(MODELS, include_paid=True)]
    assert "paid/model" in ids and "no-pricing/model" in ids


def test_main_prints_the_list(monkeypatch, capsys):
    monkeypatch.setattr(tool.httpx, "get", lambda url, timeout: httpx.Response(200, json={"data": MODELS}))
    assert tool.main([]) == 0
    output = capsys.readouterr().out
    assert "* free/json:free" in output
    assert "  free/plain:free" in output
    assert "paid/model" not in output


def test_main_reports_a_failed_request(monkeypatch, capsys):
    monkeypatch.setattr(tool.httpx, "get", lambda url, timeout: httpx.Response(500))
    assert tool.main([]) == 1
    assert "500" in capsys.readouterr().out
