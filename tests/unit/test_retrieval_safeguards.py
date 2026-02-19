from types import SimpleNamespace

from research_assistant.graph.nodes import _dedupe_results_by_url, generate_report_node
from research_assistant.services.citation import CitationManager
from research_assistant.services.formatter import ReportFormatter
from research_assistant.services.llm import LLMService


def test_dedupe_results_by_url():
    rows = [
        {"url": "https://example.com/a", "title": "a"},
        {"url": "https://example.com/a", "title": "a-dup"},
        {"url": "https://example.com/b", "title": "b"},
    ]
    out = _dedupe_results_by_url(rows)
    assert len(out) == 2
    assert out[0]["url"] == "https://example.com/a"
    assert out[1]["url"] == "https://example.com/b"


def test_generate_report_adds_source_diversity_warning():
    state = {
        "query": "Test query",
        "analyzed_content": [
            {
                "sub_question": "Q1",
                "key_information": ["finding 1"],
                "citations": ["https://same-domain.com/a", "https://same-domain.com/b"],
                "relevance_score": 0.9,
                "credibility_score": 0.8,
                "contradictions": [],
            }
        ],
        "identified_gaps": [],
        "report_sections": {"Section": "Body"},
        "metadata": {
            "settings": SimpleNamespace(min_unique_source_domains=2),
            "services": {
                "citation": CitationManager(),
                "formatter": ReportFormatter(),
                "llm": LLMService(provider="mock"),
            },
        },
    }
    out = generate_report_node(state)
    assert "Source diversity warning:" in out["final_report"]

