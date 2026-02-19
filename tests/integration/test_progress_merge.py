import time

from research_assistant.assistant import ResearchAssistant
from research_assistant.config import Settings


def test_async_progress_merges_fields(monkeypatch, tmp_path):
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        session_db_path=str(tmp_path / "sessions.db"),
    )
    assistant = ResearchAssistant(settings=settings)

    def fake_research(self, query, progress_callback=None):
        if callable(progress_callback):
            progress_callback({"stage": "plan_research_done", "sub_questions_done": 0, "sub_questions_total": 3})
            progress_callback({"stage": "analyze_content", "iteration": 1})
            progress_callback({"stage": "completed"})
        return {"final_report": "# done"}

    monkeypatch.setattr("research_assistant.assistant.ResearchAssistant.research", fake_research)
    try:
        sid = assistant.research_async("x")
        deadline = time.time() + 5
        status = "queued"
        while time.time() < deadline:
            row = assistant.get_progress(sid)
            status = row["status"]
            if status in {"completed", "failed"}:
                break
            time.sleep(0.05)
        assert status == "completed"
        progress = assistant.get_progress(sid).get("progress", {})
        assert progress.get("stage") == "completed"
        assert progress.get("sub_questions_total") == 3
        assert progress.get("sub_questions_done") == 0
        assert progress.get("iteration") == 1
    finally:
        assistant.close()
