from research_assistant.services.search import SearchService


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "organic_results": [
                {"title": "Result 1", "link": "https://example.com/1", "snippet": "Snippet 1"},
                {"title": "Result 2", "link": "https://example.com/2", "snippet": "Snippet 2"},
            ]
        }


def test_serpapi_provider_calls_serpapi_endpoint(monkeypatch):
    calls = {}

    def fake_get(url, params=None, timeout=0):
        calls["url"] = url
        calls["params"] = params
        calls["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("research_assistant.services.search.requests.get", fake_get)
    search = SearchService(provider="serpapi", serpapi_api_key="serpapi-test-key")
    items = search.search("ai in education", max_results=2)

    assert len(items) == 2
    assert calls["url"] == "https://serpapi.com/search.json"
    assert calls["params"]["api_key"] == "serpapi-test-key"
    assert calls["params"]["num"] == 2


def test_serpapi_requires_key():
    search = SearchService(provider="serpapi")
    try:
        search.search("test")
        assert False, "Expected ValueError for missing SERPAPI_API_KEY"
    except ValueError as exc:
        assert "SERPAPI_API_KEY" in str(exc)

