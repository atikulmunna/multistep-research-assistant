import requests

from research_assistant.services.llm import LLMService


class _OkResponse:
    def __init__(self, content: str):
        self._content = content
        self.status_code = 200
        self.headers = {}

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _RateLimitedResponse:
    def __init__(self):
        self.status_code = 429
        self.headers = {"Retry-After": "0"}

    def raise_for_status(self):
        raise requests.HTTPError("429 too many requests", response=self)


def test_groq_provider_calls_groq_endpoint(monkeypatch):
    calls = {}

    def fake_post(url, headers=None, json=None, timeout=0):
        calls["url"] = url
        calls["headers"] = headers
        calls["json"] = json
        return _OkResponse('{"query_type":"exploratory","scope":"broad"}')

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    llm = LLMService(provider="groq", model="llama-3.1-8b-instant", groq_api_key="gsk-test")
    result = llm.analyze_query("impact of ai on education")
    assert result["query_type"] == "exploratory"
    assert calls["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert calls["headers"]["Authorization"] == "Bearer gsk-test"


def test_provider_level_fallback_openrouter_to_groq(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=0):
        calls.append((url, json["model"]))
        if "openrouter.ai" in url:
            return _RateLimitedResponse()
        return _OkResponse('{"query_type":"exploratory","scope":"broad"}')

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    llm = LLMService(
        provider="openrouter",
        model="openai/gpt-oss-120b:free",
        openrouter_api_key="or-test",
        groq_api_key="gsk-test",
        fallback_enabled=True,
        fallback_provider="groq",
        fallback_model="llama-3.1-8b-instant",
        retry_max_attempts=1,
    )
    result = llm.analyze_query("impact of ai on education")
    assert result["query_type"] == "exploratory"
    assert any("openrouter.ai" in c[0] for c in calls)
    assert any("api.groq.com" in c[0] for c in calls)

