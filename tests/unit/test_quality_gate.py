from types import SimpleNamespace

from research_assistant.graph.nodes import generate_report_node
from research_assistant.services.citation import CitationManager
from research_assistant.services.formatter import ReportFormatter


class _FakeLLM:
    def summarize_report(self, query, sections):
        return "summary"


def test_generate_report_quality_metadata_includes_failed_checks():
    state = {
        "query": "test query",
        "report_sections": {"Findings Overview": "content"},
        "analyzed_content": [
            {
                "sub_question": "q1",
                "key_information": ["fact"],
                "citations": ["https://example.com/a"],
            }
        ],
        "identified_gaps": [],
        "metadata": {
            "services": {
                "llm": _FakeLLM(),
                "citation": CitationManager(),
                "formatter": ReportFormatter(),
            },
            "settings": SimpleNamespace(min_unique_source_domains=2, min_reference_count=3),
        },
    }

    out = generate_report_node(state)
    quality = out["metadata"]["quality"]
    assert quality["passed"] is False
    assert "reference_count_below_min" in quality["checks_failed"]
    assert "source_diversity_below_min" in quality["checks_failed"]
