import json
import time

from typer.testing import CliRunner

from research_assistant.assistant import ResearchAssistant
from research_assistant.config import Settings
from research_assistant.main import app


def test_metrics_command_json_output():
    runner = CliRunner()
    result = runner.invoke(app, ["metrics", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "llm" in payload
    assert "ops" in payload


def test_metrics_reset_clears_history(tmp_path, monkeypatch):
    env = {
        "SESSION_DB_PATH": str(tmp_path / "sessions.db"),
        "LLM_PROVIDER": "mock",
        "SEARCH_PROVIDER": "mock",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    assistant = ResearchAssistant(settings=Settings())
    try:
        sid = assistant.research_async("Impact of AI on education")
        deadline = time.time() + 5
        status = "queued"
        while time.time() < deadline:
            status = assistant.get_progress(sid)["status"]
            if status in {"completed", "failed"}:
                break
            time.sleep(0.05)
        assert status == "completed"
    finally:
        assistant.close()

    runner = CliRunner()

    before = runner.invoke(app, ["metrics", "--json"])
    assert before.exit_code == 0
    before_payload = json.loads(before.stdout)
    assert before_payload["ops"]["session_counts"].get("completed", 0) >= 1

    reset = runner.invoke(app, ["metrics", "--reset", "--yes", "--json"])
    assert reset.exit_code == 0
    reset_payload = json.loads(reset.stdout)
    assert reset_payload["reset"]["llm_metrics_cleared"] is True

    after = runner.invoke(app, ["metrics", "--json"])
    assert after.exit_code == 0
    after_payload = json.loads(after.stdout)
    assert after_payload["ops"]["session_counts"] == {}
