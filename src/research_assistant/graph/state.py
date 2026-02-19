from typing import Dict, List, TypedDict


class ResearchState(TypedDict):
    query: str
    analyzed_query: Dict
    sub_questions: List[Dict]
    current_question_idx: int

    search_results: Dict[str, List[Dict]]
    raw_documents: Dict[str, List[Dict]]

    analyzed_content: List[Dict]
    identified_gaps: List[str]
    contradictions: List[Dict]

    report_sections: Dict[str, str]
    final_report: str

    metadata: Dict
    errors: List[str]
    total_tokens_used: int
    execution_time: float
    iteration_count: int

