import json
import time

from typer.testing import CliRunner

from research_assistant.assistant import ResearchAssistant
from research_assistant.config import Settings
from research_assistant.main import app


def _seed_completed_session(db_path: str, reports_dir: str) -> str:
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        session_db_path=db_path,
        reports_directory=reports_dir,
    )
    assistant = ResearchAssistant(settings=settings)
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
        return sid
    finally:
        assistant.close()


def test_reports_list_show_and_export(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "sessions.db"
    _seed_completed_session(str(db_path), str(reports_dir))

    (reports_dir / "latest_report.md").write_text("# File Report\n\ncontent", encoding="utf-8")

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    monkeypatch.setenv("SESSION_DB_PATH", str(db_path))
    monkeypatch.setenv("REPORTS_DIRECTORY", str(reports_dir))

    runner = CliRunner()

    list_result = runner.invoke(app, ["reports", "list", "--json"])
    assert list_result.exit_code == 0
    list_payload = json.loads(list_result.stdout)
    assert list_payload["count"] >= 1
    assert any(row["ref"].startswith("session:") or row["ref"].startswith("file:") for row in list_payload["entries"])

    show_result = runner.invoke(app, ["reports", "show", "latest", "--json"])
    assert show_result.exit_code == 0
    show_payload = json.loads(show_result.stdout)
    assert show_payload["markdown"]

    out_html = reports_dir / "export_test.html"
    export_result = runner.invoke(
        app,
        ["reports", "export", "--from", "latest", "--to", "html", "--output", str(out_html)],
    )
    assert export_result.exit_code == 0
    assert out_html.exists()
    body = out_html.read_text(encoding="utf-8")
    assert "<html>" in body.lower()

    out_pdf = reports_dir / "export_test.pdf"
    export_pdf_result = runner.invoke(
        app,
        ["reports", "export", "--from", "latest", "--to", "pdf", "--output", str(out_pdf)],
    )
    assert export_pdf_result.exit_code == 0
    assert out_pdf.exists()
    assert out_pdf.read_bytes().startswith(b"%PDF")


def test_reports_show_preview_and_full(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "sessions.db"
    lines = [f"line {i}" for i in range(1, 31)]
    (reports_dir / "preview.md").write_text("\n".join(lines), encoding="utf-8")

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    monkeypatch.setenv("SESSION_DB_PATH", str(db_path))
    monkeypatch.setenv("REPORTS_DIRECTORY", str(reports_dir))

    runner = CliRunner()
    preview_result = runner.invoke(app, ["reports", "show", "file:preview.md", "--preview-lines", "10"])
    assert preview_result.exit_code == 0
    assert "truncated preview" in preview_result.stdout
    assert "line 10" in preview_result.stdout
    assert "line 30" not in preview_result.stdout

    full_result = runner.invoke(app, ["reports", "show", "file:preview.md", "--full"])
    assert full_result.exit_code == 0
    assert "line 30" in full_result.stdout


def test_reports_latest_prefers_file_with_kind_filter(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "sessions.db"
    _seed_completed_session(str(db_path), str(reports_dir))
    file_report = reports_dir / "latest_report.md"
    file_report.write_text("# File Preferred\n\nreal file report", encoding="utf-8")

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    monkeypatch.setenv("SESSION_DB_PATH", str(db_path))
    monkeypatch.setenv("REPORTS_DIRECTORY", str(reports_dir))

    runner = CliRunner()
    latest_default = runner.invoke(app, ["reports", "show", "latest"])
    assert latest_default.exit_code == 0
    assert "File Preferred" in latest_default.stdout

    latest_session = runner.invoke(app, ["reports", "show", "latest", "--kind", "session"])
    assert latest_session.exit_code == 0
    assert "Research Report:" in latest_session.stdout

    list_files = runner.invoke(app, ["reports", "list", "--kind", "file", "--json"])
    assert list_files.exit_code == 0
    payload = json.loads(list_files.stdout)
    assert payload["count"] >= 1
    assert all(row["kind"] == "file" for row in payload["entries"])
