from typer.testing import CliRunner

from research_assistant.main import app


def test_watch_command_streams_and_exits(monkeypatch):
    states = [
        {"session_id": "s1", "status": "running", "progress": {"stage": "analyze_query"}},
        {
            "session_id": "s1",
            "status": "running",
            "progress": {"stage": "gather_information", "sub_questions_done": 1, "sub_questions_total": 3},
        },
        {"session_id": "s1", "status": "completed", "progress": {"stage": "completed"}},
    ]
    calls = {"n": 0}

    def fake_get_progress(self, session_id):
        idx = min(calls["n"], len(states) - 1)
        calls["n"] += 1
        return states[idx]

    monkeypatch.setattr("research_assistant.main.ResearchAssistant.get_progress", fake_get_progress)
    monkeypatch.setattr("research_assistant.main.time.sleep", lambda _: None)

    runner = CliRunner()
    result = runner.invoke(app, ["watch", "s1", "--interval", "0.1", "--timeout", "2"])
    assert result.exit_code == 0
    assert "status=running stage=analyze_query" in result.stdout
    assert "status=running stage=gather_information sub_questions=1/3" in result.stdout
    assert "status=completed stage=completed" in result.stdout
