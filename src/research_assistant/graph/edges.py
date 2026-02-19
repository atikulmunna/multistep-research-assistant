from typing import Dict, List

from .state import ResearchState


def should_continue_gathering(state: ResearchState) -> str:
    if state["current_question_idx"] < len(state["sub_questions"]):
        return "gather"
    return "analyze"


def is_information_sufficient(state: ResearchState) -> str:
    settings = state["metadata"]["settings"]
    coverage = calculate_coverage(state["analyzed_content"], settings.min_relevance_score)
    gaps = identify_critical_gaps(state["analyzed_content"], settings.min_relevance_score)

    if coverage >= settings.min_relevance_score and not gaps:
        return "synthesize"

    if state["iteration_count"] >= settings.max_research_iterations:
        warnings = list(state["metadata"].get("warnings", []))
        warnings.append("Iteration cap reached; continuing with partial evidence.")
        state["metadata"]["warnings"] = warnings
        return "synthesize"

    if gaps:
        state["sub_questions"].extend(generate_gap_questions(gaps, state["sub_questions"]))
    return "gather"


def calculate_coverage(analyzed_content: List[Dict], min_relevance: float) -> float:
    if not analyzed_content:
        return 0.0
    covered = sum(1 for item in analyzed_content if item.get("relevance_score", 0.0) >= min_relevance)
    return covered / len(analyzed_content)


def identify_critical_gaps(analyzed_content: List[Dict], min_relevance: float) -> List[str]:
    gaps = []
    for item in analyzed_content:
        if item.get("relevance_score", 0.0) < min_relevance:
            gaps.append(item["sub_question"])
    return gaps


def generate_gap_questions(gaps: List[str], existing_questions: List[Dict]) -> List[Dict]:
    existing = {row["question"] for row in existing_questions}
    generated: List[Dict] = []
    for gap in gaps:
        candidate = f"Additional evidence for {gap}"
        if candidate not in existing:
            generated.append({"question": candidate, "priority": 5, "answered": False, "parent_question": gap})
            existing.add(candidate)
    return generated

