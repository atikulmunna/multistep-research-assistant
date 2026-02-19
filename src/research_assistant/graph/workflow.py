from langgraph.graph import END, StateGraph

from .edges import is_information_sufficient, should_continue_gathering
from .nodes import (
    analyze_content_node,
    analyze_query_node,
    gather_information_node,
    generate_report_node,
    plan_research_node,
    synthesize_information_node,
)
from .state import ResearchState


def create_research_workflow():
    workflow = StateGraph(ResearchState)
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("plan_research", plan_research_node)
    workflow.add_node("gather_information", gather_information_node)
    workflow.add_node("analyze_content", analyze_content_node)
    workflow.add_node("synthesize", synthesize_information_node)
    workflow.add_node("generate_report", generate_report_node)

    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "plan_research")
    workflow.add_conditional_edges(
        "plan_research",
        should_continue_gathering,
        {"gather": "gather_information", "analyze": "analyze_content"},
    )
    workflow.add_edge("gather_information", "analyze_content")
    workflow.add_conditional_edges(
        "analyze_content",
        is_information_sufficient,
        {"gather": "gather_information", "synthesize": "synthesize"},
    )
    workflow.add_edge("synthesize", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()

