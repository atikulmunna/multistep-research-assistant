from research_assistant.services.llm import LLMService


class _FakeOllamaResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "message": {"role": "assistant", "content": '{"query_type":"exploratory","scope":"broad"}'},
            "prompt_eval_count": 12,
            "eval_count": 8,
        }


def test_ollama_provider_calls_local_endpoint(monkeypatch):
    calls = {}

    def fake_post(url, headers=None, json=None, timeout=0):
        calls["url"] = url
        calls["headers"] = headers
        calls["json"] = json
        calls["timeout"] = timeout
        return _FakeOllamaResponse()

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    llm = LLMService(
        provider="ollama",
        model="llama3.1:8b",
        ollama_base_url="http://127.0.0.1:11434",
    )
    result = llm.analyze_query("impact of ai on education")
    assert result["query_type"] == "exploratory"
    assert calls["url"] == "http://127.0.0.1:11434/api/chat"
    assert calls["json"]["model"] == "llama3.1:8b"
    assert calls["json"]["stream"] is False
