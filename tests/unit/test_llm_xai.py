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


def test_xai_provider_calls_xai_endpoint(monkeypatch):
    calls = {}

    def fake_post(url, headers=None, json=None, timeout=0):
        calls["url"] = url
        calls["headers"] = headers
        calls["json"] = json
        calls["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    llm = LLMService(provider="xai", model="grok-4", xai_api_key="xai-test-key")
    result = llm.analyze_query("impact of ai on education")

    assert result["query_type"] == "exploratory"
    assert calls["url"] == "https://api.x.ai/v1/chat/completions"
    assert calls["headers"]["Authorization"] == "Bearer xai-test-key"
    assert calls["json"]["model"] == "grok-4"


def test_grok_alias_requires_key():
    llm = LLMService(provider="grok", model="grok-4")
    try:
        llm.analyze_query("test")
        assert False, "Expected ValueError for missing XAI_API_KEY"
    except ValueError as exc:
        assert "XAI_API_KEY" in str(exc)

