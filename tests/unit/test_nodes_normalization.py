from types import SimpleNamespace

from research_assistant.graph.nodes import analyze_content_node


class _FakeLLM:
    def extract_key_info(self, sub_question, docs):
        return {
            "key_information": ["Valid finding"],
            "citations": [{"url": "https://example.com/source"}, {"link": "https://example.org/ref"}],
            "contradictions": [],
        }


def test_analyze_content_normalizes_dict_citations():
    state = {
        "sub_questions": [{"question": "Q1", "priority": 1, "answered": False, "parent_question": None}],
        "raw_documents": {"Q1": []},
        "analyzed_content": [],
        "contradictions": [],
        "metadata": {
            "services": {"llm": _FakeLLM()},
            "settings": SimpleNamespace(min_relevance_score=0.8),
        },
        "iteration_count": 0,
    }

    out = analyze_content_node(state)
    assert len(out["analyzed_content"]) == 1
    item = out["analyzed_content"][0]
    assert item["citations"] == ["https://example.com/source", "https://example.org/ref"]
    assert 0.5 <= item["credibility_score"] <= 0.95


class _FakeLLMPlaceholders:
    def extract_key_info(self, sub_question, docs):
        return {
            "key_information": ["Valid finding"],
            "citations": ["Source 1", "[citation 1]"],
            "contradictions": [],
        }


def test_analyze_content_filters_placeholder_citations_and_falls_back_to_docs():
    state = {
        "sub_questions": [{"question": "Q1", "priority": 1, "answered": False, "parent_question": None}],
        "raw_documents": {
            "Q1": [
                {"source_url": "https://example.com/a"},
                {"source_url": "https://example.org/b"},
            ]
        },
        "analyzed_content": [],
        "contradictions": [],
        "metadata": {
            "services": {"llm": _FakeLLMPlaceholders()},
            "settings": SimpleNamespace(min_relevance_score=0.8),
        },
        "iteration_count": 0,
    }

    out = analyze_content_node(state)
    item = out["analyzed_content"][0]
    # Model placeholders are removed; source URLs are used as fallback citations.
    assert item["citations"] == ["https://example.com/a", "https://example.org/b"]
    assert "Source 1" not in item["citations"]
    assert "[citation 1]" not in item["citations"]
