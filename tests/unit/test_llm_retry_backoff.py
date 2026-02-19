import requests

from research_assistant.services.llm import LLMService


class _RateLimitedResponse:
    def __init__(self, retry_after: str = "0"):
        self.status_code = 429
        self.headers = {"Retry-After": retry_after}
        self.text = "rate limited"

    def raise_for_status(self):
        raise requests.HTTPError("429 too many requests", response=self)


class _BadRequestResponse:
    def __init__(self):
        self.status_code = 400
        self.headers = {}
        self.text = "bad request"

    def raise_for_status(self):
        raise requests.HTTPError("400 bad request", response=self)


class _SuccessResponse:
    def __init__(self):
        self.status_code = 200
        self.headers = {}

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": '{"query_type":"exploratory","scope":"broad"}'}}]}


def test_openrouter_retries_on_429_then_succeeds(monkeypatch):
    calls = {"count": 0}
    sleeps = []

    def fake_post(url, headers=None, json=None, timeout=0):
        calls["count"] += 1
        if calls["count"] < 3:
            return _RateLimitedResponse(retry_after="0")
        return _SuccessResponse()

    def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    monkeypatch.setattr("research_assistant.services.llm.time.sleep", fake_sleep)

    llm = LLMService(
        provider="openrouter",
        model="openai/gpt-oss-120b:free",
        openrouter_api_key="or-test-key",
        retry_max_attempts=4,
        retry_base_delay_s=0.01,
        retry_max_delay_s=0.1,
    )
    result = llm.analyze_query("impact of ai on education")

    assert result["query_type"] == "exploratory"
    assert calls["count"] == 3
    assert len(sleeps) == 2


def test_openrouter_does_not_retry_400(monkeypatch):
    calls = {"count": 0}

    def fake_post(url, headers=None, json=None, timeout=0):
        calls["count"] += 1
        return _BadRequestResponse()

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    llm = LLMService(
        provider="openrouter",
        model="openai/gpt-oss-120b:free",
        openrouter_api_key="or-test-key",
        retry_max_attempts=4,
    )
    try:
        llm.analyze_query("impact of ai on education")
        assert False, "Expected HTTPError for 400 without retries"
    except requests.HTTPError:
        pass
    assert calls["count"] == 1

