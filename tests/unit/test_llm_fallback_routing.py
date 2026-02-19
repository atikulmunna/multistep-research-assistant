import requests

from research_assistant.services.llm import LLMService


class _FakeResponse:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_route_fallback_uses_base_model_and_records_metrics(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=0):
        model = json["model"]
        calls.append(model)
        if model == "bad-planning-model":
            raise requests.HTTPError("bad model")
        return _FakeResponse('{"query_type":"exploratory","scope":"broad"}')

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)

    llm = LLMService(
        provider="openrouter",
        model="openai/gpt-oss-120b:free",
        openrouter_api_key="or-test",
        task_models={"planning": "bad-planning-model"},
        fallback_enabled=True,
        fallback_model="openai/gpt-oss-120b:free",
    )

    result = llm.analyze_query("impact of ai on education")
    assert result["query_type"] == "exploratory"
    assert calls == ["bad-planning-model", "openai/gpt-oss-120b:free"]

    metrics = llm.get_metrics()
    routed_key = "planning:openrouter:openai/gpt-oss-120b:free"
    assert routed_key in metrics
    assert metrics[routed_key]["fallback_calls"] == 1
    assert metrics[routed_key]["successes"] == 1


def test_route_fallback_disabled_raises(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=0):
        raise requests.HTTPError("bad model")

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    llm = LLMService(
        provider="openrouter",
        model="openai/gpt-oss-120b:free",
        openrouter_api_key="or-test",
        task_models={"planning": "bad-planning-model"},
        fallback_enabled=False,
    )

    try:
        llm.analyze_query("impact of ai on education")
        assert False, "Expected requests.HTTPError when fallback disabled"
    except requests.HTTPError:
        pass


def test_second_fallback_provider_used_after_first_fallback_fails(monkeypatch):
    calls = []

    class _FakeOllamaResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"query_type":"exploratory","scope":"broad"}'}}

    def fake_post(url, headers=None, json=None, timeout=0):
        calls.append((url, json.get("model")))
        if "openrouter.ai" in url:
            raise requests.HTTPError("primary failed")
        if "api.groq.com" in url:
            raise requests.HTTPError("first fallback failed")
        return _FakeOllamaResponse()

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    llm = LLMService(
        provider="openrouter",
        model="openai/gpt-oss-120b:free",
        openrouter_api_key="or-test",
        groq_api_key="gsk-test",
        fallback_enabled=True,
        fallback_provider="groq",
        fallback_model="llama-3.1-8b-instant",
        second_fallback_provider="ollama",
        second_fallback_model="llama3.1:8b",
        retry_max_attempts=1,
    )
    result = llm.analyze_query("impact of ai on education")
    assert result["query_type"] == "exploratory"
    assert len(calls) == 3
    assert "openrouter.ai" in calls[0][0]
    assert "api.groq.com" in calls[1][0]
    assert calls[2][0].endswith("/api/chat")
