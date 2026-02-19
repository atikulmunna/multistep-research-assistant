from research_assistant.services.llm import LLMService


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"query_type":"exploratory","scope":"broad"}'
                    }
                }
            ]
        }


def test_openrouter_provider_calls_openrouter_endpoint(monkeypatch):
    calls = {}

    def fake_post(url, headers=None, json=None, timeout=0):
        calls["url"] = url
        calls["headers"] = headers
        calls["json"] = json
        calls["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    llm = LLMService(
        provider="openrouter",
        model="openai/gpt-oss-20b:free",
        openrouter_api_key="or-test-key",
    )
    result = llm.analyze_query("impact of ai on education")

    assert result["query_type"] == "exploratory"
    assert calls["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert calls["headers"]["Authorization"] == "Bearer or-test-key"
    assert calls["json"]["model"] == "openai/gpt-oss-20b:free"


def test_openrouter_requires_key():
    llm = LLMService(provider="openrouter", model="openai/gpt-oss-20b:free")
    try:
        llm.analyze_query("test")
        assert False, "Expected ValueError for missing OPENROUTER_API_KEY"
    except ValueError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)

