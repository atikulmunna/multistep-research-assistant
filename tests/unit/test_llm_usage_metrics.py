from research_assistant.services.llm import LLMService


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": '{"query_type":"exploratory","scope":"broad"}'}}],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "total_cost": 0.00021,
            },
        }


def test_llm_usage_metrics_capture_tokens_and_cost(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=0):
        return _FakeResponse()

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    llm = LLMService(
        provider="openrouter",
        model="openai/gpt-oss-120b:free",
        openrouter_api_key="or-test",
    )
    llm.analyze_query("impact of ai on education")
    metrics = llm.get_metrics()
    key = "planning:openrouter:openai/gpt-oss-120b:free"
    assert key in metrics
    assert metrics[key]["prompt_tokens"] == 120
    assert metrics[key]["completion_tokens"] == 30
    assert metrics[key]["total_tokens"] == 150
    assert abs(metrics[key]["total_cost_usd"] - 0.00021) < 1e-12
