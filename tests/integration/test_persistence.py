import time

from research_assistant.assistant import ResearchAssistant
from research_assistant.config import Settings


def test_session_persists_across_assistant_instances(tmp_path):
    db_path = tmp_path / "sessions.db"
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        max_research_iterations=5,
        min_relevance_score=0.8,
        session_db_path=str(db_path),
    )

    assistant1 = ResearchAssistant(settings=settings)
    session_id = assistant1.research_async("Impact of AI on education")

    deadline = time.time() + 5
    status = "queued"
    while time.time() < deadline:
        progress = assistant1.get_progress(session_id)
        status = progress["status"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert status == "completed"
    assistant1.close()

    assistant2 = ResearchAssistant(settings=settings)
    restored_progress = assistant2.get_progress(session_id)
    assert restored_progress["status"] == "completed"
    restored_result = assistant2.get_result(session_id)
    assert restored_result["result"]["final_report"]
    assistant2.close()

