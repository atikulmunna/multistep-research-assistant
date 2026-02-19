import requests

from research_assistant.graph.nodes import gather_information_node


class _SearchSvc:
    def search(self, query, max_results=5):
        return [
            {"url": "https://bad.example.com", "title": "bad"},
            {"url": "mock://good/1", "title": "good"},
        ]

    def fetch_url(self, url):
        if "bad.example.com" in url:
            raise requests.HTTPError("403", response=type("R", (), {"status_code": 403})())
        return "plain text content for good source"


class _ParserSvc:
    def detect_type(self, url, content):
        return "txt"

    def parse(self, content, doc_type, source_url):
        return {"content": content, "source_url": source_url, "doc_type": doc_type, "metadata": {}}


def test_gather_information_skips_inaccessible_urls():
    state = {
        "sub_questions": [{"question": "q1"}],
        "current_question_idx": 0,
        "search_results": {},
        "raw_documents": {},
        "errors": [],
        "metadata": {
            "warnings": [],
            "settings": type("S", (), {"max_search_results": 5})(),
            "services": {"search": _SearchSvc(), "parser": _ParserSvc()},
        },
    }

    out = gather_information_node(state)
    assert out["current_question_idx"] == 1
    assert "q1" in out["raw_documents"]
    assert len(out["raw_documents"]["q1"]) == 1
    assert any("fetch_failed:https://bad.example.com" in e for e in out["errors"])
    assert any("Skipping inaccessible URL: https://bad.example.com" in w for w in out["metadata"]["warnings"])

