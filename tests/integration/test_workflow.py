import time

from research_assistant.assistant import ResearchAssistant
from research_assistant.config import Settings


def test_complete_workflow_mock():
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        max_research_iterations=5,
        min_relevance_score=0.8,
    )
    assistant = ResearchAssistant(settings=settings)
    result = assistant.research("Benefits of renewable energy")
    report = result["final_report"]

    assert report
    assert "## Executive Summary" in report
    assert "## References" in report
    assert result["current_question_idx"] > 0


def test_async_workflow_progress_and_result():
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        max_research_iterations=5,
        min_relevance_score=0.8,
    )
    assistant = ResearchAssistant(settings=settings)
    session_id = assistant.research_async("Impact of AI on education")

    deadline = time.time() + 5
    status = "queued"
    while time.time() < deadline:
        progress = assistant.get_progress(session_id)
        status = progress["status"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert status == "completed"
    result = assistant.get_result(session_id)
    assert result["result"]["final_report"]
