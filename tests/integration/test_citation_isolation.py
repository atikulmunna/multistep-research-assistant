from research_assistant.assistant import ResearchAssistant
from research_assistant.config import Settings


def _slug(text: str) -> str:
    return text.replace(" ", "_")


def test_citations_do_not_leak_between_runs(tmp_path):
    reports_dir = tmp_path / "reports"
    db_path = tmp_path / "sessions.db"
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        reports_directory=str(reports_dir),
        session_db_path=str(db_path),
        max_search_results=1,
        max_sub_questions=1,
        max_research_iterations=1,
    )
    assistant = ResearchAssistant(settings=settings)
    try:
        llm = assistant.services["llm"]
        llm.decompose_query = lambda query, max_questions=5: [  # type: ignore[assignment]
            {"question": query, "priority": 1, "answered": False, "parent_question": None}
        ]

        q1 = "Electric vehicle policy in India and US"
        q2 = "Progress in physical AI since NVIDIA stepped in"
        r1 = assistant.research(q1)["final_report"]
        r2 = assistant.research(q2)["final_report"]

        url1 = f"mock://{_slug(q1)}/1"
        url2 = f"mock://{_slug(q2)}/1"
        assert url1 in r1
        assert url2 in r2
        assert url1 not in r2
    finally:
        assistant.close()
