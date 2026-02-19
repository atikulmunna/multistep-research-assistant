from research_assistant.services.llm import LLMService


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_chat_json_parses_markdown_wrapped_json(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=0):
        return _FakeResponse("```json\n{\"query_type\":\"exploratory\",\"scope\":\"broad\"}\n```")

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    llm = LLMService(provider="openrouter", model="openai/gpt-oss-120b:free", openrouter_api_key="or-test")
    result = llm.analyze_query("impact of ai on education")
    assert result["query_type"] == "exploratory"
    assert result["scope"] == "broad"


def test_chat_json_repairs_invalid_output_with_second_call(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse("Here is your answer:\nquery_type=exploratory; scope=broad")
        return _FakeResponse('{"query_type":"exploratory","scope":"broad"}')

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    llm = LLMService(provider="openrouter", model="openai/gpt-oss-120b:free", openrouter_api_key="or-test")
    result = llm.analyze_query("impact of ai on education")
    assert result["query_type"] == "exploratory"
    assert calls["n"] == 2


def test_chat_json_uses_second_repair_pass(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse("not json at all")
        if calls["n"] == 2:
            return _FakeResponse("{query_type: exploratory, scope: broad}")
        return _FakeResponse('{"query_type":"exploratory","scope":"broad"}')

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    llm = LLMService(provider="openrouter", model="openai/gpt-oss-120b:free", openrouter_api_key="or-test")
    result = llm.analyze_query("impact of ai on education")
    assert result["scope"] == "broad"
    assert calls["n"] == 3


def test_chat_json_rejects_python_ellipsis_values(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse("{'query_type':'exploratory','scope': ...}")
        return _FakeResponse('{"query_type":"exploratory","scope":"broad"}')

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    llm = LLMService(provider="openrouter", model="openai/gpt-oss-120b:free", openrouter_api_key="or-test")
    result = llm.analyze_query("impact of ai on education")
    assert result["scope"] == "broad"
    assert calls["n"] == 2


def test_decompose_query_falls_back_when_sub_questions_missing(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=0):
        return _FakeResponse('{"plan":"ok"}')

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    llm = LLMService(provider="openrouter", model="openai/gpt-oss-120b:free", openrouter_api_key="or-test")
    result = llm.decompose_query("impact of ai on education", max_questions=3)
    assert isinstance(result, list)
    assert len(result) == 3
    assert all("question" in row for row in result)
