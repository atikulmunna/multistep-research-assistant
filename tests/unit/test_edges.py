from types import SimpleNamespace

from research_assistant.graph.edges import calculate_coverage, is_information_sufficient


def test_calculate_coverage():
    rows = [
        {"sub_question": "q1", "relevance_score": 0.9},
        {"sub_question": "q2", "relevance_score": 0.7},
    ]
    assert calculate_coverage(rows, 0.8) == 0.5


def test_is_information_sufficient_returns_synthesize():
    state = {
        "analyzed_content": [
            {"sub_question": "q1", "relevance_score": 0.92},
            {"sub_question": "q2", "relevance_score": 0.88},
        ],
        "metadata": {"settings": SimpleNamespace(min_relevance_score=0.8, max_research_iterations=5), "warnings": []},
        "iteration_count": 1,
        "sub_questions": [{"question": "q1"}, {"question": "q2"}],
    }
    assert is_information_sufficient(state) == "synthesize"


def test_is_information_sufficient_adds_gap_questions():
    state = {
        "analyzed_content": [{"sub_question": "q1", "relevance_score": 0.5}],
        "metadata": {"settings": SimpleNamespace(min_relevance_score=0.8, max_research_iterations=5), "warnings": []},
        "iteration_count": 1,
        "sub_questions": [{"question": "q1", "priority": 1, "answered": False, "parent_question": None}],
    }
    decision = is_information_sufficient(state)
    assert decision == "gather"
    assert any("Additional evidence for q1" == row["question"] for row in state["sub_questions"])

