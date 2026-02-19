from contextlib import asynccontextmanager
from pathlib import Path
import time
from threading import Lock
import json
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from .assistant import ResearchAssistant
from .config import Settings


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)


def create_app(assistant: ResearchAssistant | None = None) -> FastAPI:
    managed_assistant = assistant or ResearchAssistant(settings=Settings())
    settings = managed_assistant.settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            app.state.assistant.close()

    app = FastAPI(title="Research Assistant API", version="0.1.0", lifespan=lifespan)
    app.state.assistant = managed_assistant
    app.state.rate_lock = Lock()
    app.state.rate_counters = {}

    @app.middleware("http")
    async def auth_and_rate_limit(request: Request, call_next):
        if not request.url.path.startswith("/api/v1/"):
            return await call_next(request)

        auth_token = settings.api_auth_token.strip()
        provided = request.headers.get("x-api-key", "")
        if auth_token and provided != auth_token:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        per_minute = int(settings.api_rate_limit_per_minute or 0)
        if per_minute > 0:
            identifier = provided or (request.client.host if request.client else "anonymous")
            current_window = int(time.time() // 60)
            with app.state.rate_lock:
                bucket = app.state.rate_counters.get(identifier)
                if not bucket or bucket["window"] != current_window:
                    bucket = {"window": current_window, "count": 0}
                if bucket["count"] >= per_minute:
                    app.state.rate_counters[identifier] = bucket
                    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
                bucket["count"] += 1
                app.state.rate_counters[identifier] = bucket

        return await call_next(request)

    @app.post("/api/v1/research", status_code=202)
    def start_research(payload: ResearchRequest):
        session_id = app.state.assistant.research_async(payload.query)
        return {"session_id": session_id, "status": "queued"}

    @app.get("/api/v1/research/{session_id}/status")
    def get_status(session_id: str):
        progress = app.state.assistant.get_progress(session_id)
        if progress["status"] == "not_found":
            raise HTTPException(status_code=404, detail="Session not found")
        return progress

    @app.get("/api/v1/research/{session_id}/result")
    def get_result(session_id: str):
        result = app.state.assistant.get_result(session_id)
        if result["status"] == "not_found":
            raise HTTPException(status_code=404, detail="Session not found")
        if result["status"] != "completed":
            raise HTTPException(status_code=409, detail=f"Session not complete: {result['status']}")
        return result

    @app.post("/api/v1/research/{session_id}/cancel")
    def cancel(session_id: str):
        result = app.state.assistant.cancel_session(session_id)
        if result["status"] == "not_found":
            raise HTTPException(status_code=404, detail="Session not found")
        return result

    @app.get("/api/v1/metrics/llm")
    def get_llm_metrics():
        return {"metrics": app.state.assistant.get_llm_metrics()}

    @app.get("/api/v1/metrics/ops")
    def get_ops_metrics():
        return app.state.assistant.get_ops_metrics()

    @app.get("/api/v1/reports")
    def list_reports(limit: int = 20, kind: str = "all"):
        return {"entries": _report_entries(app.state.assistant, settings, limit=limit, kind=kind)}

    @app.get("/api/v1/reports/content")
    def get_report_content(ref: str = "latest", kind: str = "all"):
        return _resolve_report_ref(app.state.assistant, settings, ref=ref, kind=kind)

    @app.get("/api/v1/benchmarks/history")
    def get_benchmark_history(limit: int = 20):
        return {"entries": _load_benchmark_history(settings, limit=limit)}

    @app.get("/api/v1/benchmarks/compare")
    def get_benchmark_compare(file_a: str = "", file_b: str = ""):
        try:
            path_a, path_b = _resolve_compare_paths(settings, file_a=file_a, file_b=file_b)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        payload_a = json.loads(path_a.read_text(encoding="utf-8-sig"))
        payload_b = json.loads(path_b.read_text(encoding="utf-8-sig"))
        summary_a = payload_a.get("summary", {})
        summary_b = payload_b.get("summary", {})
        return {
            "file_a": str(path_a),
            "file_b": str(path_b),
            "summary_a": summary_a,
            "summary_b": summary_b,
            "delta_b_minus_a": _summary_delta(summary_a, summary_b),
        }

    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        html_path = Path(__file__).parent / "web" / "dashboard.html"
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="Dashboard file not found")
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    return app


app = create_app()


def _report_entries(assistant: ResearchAssistant, settings: Settings, limit: int, kind: str) -> list[dict]:
    k = (kind or "all").strip().lower()
    if k not in {"all", "file", "session"}:
        raise HTTPException(status_code=400, detail="Invalid kind. Use one of: all, file, session.")
    session_entries = assistant.list_completed_reports(limit=limit)
    reports_dir = Path(settings.reports_directory)
    file_entries: list[dict] = []
    if reports_dir.exists():
        md_files = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        for path in md_files:
            updated_at = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(path.stat().st_mtime))
            file_entries.append(
                {
                    "ref": f"file:{path.name}",
                    "kind": "file",
                    "session_id": "",
                    "query": path.stem,
                    "updated_at": updated_at,
                    "report_chars": int(path.stat().st_size),
                    "has_report": True,
                    "path": str(path),
                }
            )
    if k == "file":
        entries = file_entries
    elif k == "session":
        entries = session_entries
    else:
        entries = file_entries + session_entries
    if k == "all":
        entries.sort(
            key=lambda row: (
                -_iso_to_epoch(row.get("updated_at", "")),
            ),
        )
    else:
        entries.sort(key=lambda row: row.get("updated_at", ""), reverse=True)
    return entries[:limit]


def _resolve_report_ref(assistant: ResearchAssistant, settings: Settings, ref: str, kind: str) -> dict:
    target = (ref or "latest").strip()
    if target in {"", "latest"}:
        entries = _report_entries(assistant, settings, limit=1, kind=kind)
        if not entries:
            raise HTTPException(status_code=404, detail="No reports found")
        target = entries[0]["ref"]

    if target.startswith("session:"):
        sid = target.split(":", 1)[1].strip()
        row = assistant.get_result(sid)
        if row.get("status") != "completed":
            raise HTTPException(status_code=404, detail="Session report not found")
        result = row.get("result", {}) or {}
        markdown = str(result.get("final_report", "") or "")
        if not markdown:
            raise HTTPException(status_code=404, detail="Session report not found")
        return {"ref": f"session:{sid}", "kind": "session", "query": result.get("query", ""), "markdown": markdown}

    if target.startswith("file:"):
        name = target.split(":", 1)[1].strip()
        path = Path(settings.reports_directory) / name
        if not path.exists():
            raise HTTPException(status_code=404, detail="File report not found")
        return {
            "ref": f"file:{name}",
            "kind": "file",
            "query": path.stem,
            "markdown": path.read_text(encoding="utf-8"),
            "path": str(path),
        }
    raise HTTPException(status_code=400, detail="Unsupported ref. Use latest, session:<id>, or file:<name>.")


def _load_benchmark_history(settings: Settings, limit: int) -> list[dict]:
    reports_dir = Path(settings.reports_directory)
    if not reports_dir.exists():
        return []
    files = sorted(
        [p for p in reports_dir.glob("benchmark_*.json") if p.name != "benchmark_latest.json"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    entries = []
    for path in files[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        summary = payload.get("summary", {})
        entries.append(
            {
                "file": str(path),
                "timestamp": path.stem.replace("benchmark_", ""),
                "queries_run": int(summary.get("queries_run", 0)),
                "successes": int(summary.get("successes", 0)),
                "failures": int(summary.get("failures", 0)),
                "quality_gate_failures": int(summary.get("quality_gate_failures", 0)),
                "budget_failures": int(summary.get("budget_failures", 0)),
                "avg_latency_s": float(summary.get("avg_latency_s", 0.0)),
                "avg_reference_count": float(summary.get("avg_reference_count", 0.0)),
                "total_tokens": int(summary.get("total_tokens", 0)),
            }
        )
    return entries


def _resolve_compare_paths(settings: Settings, file_a: str, file_b: str) -> tuple[Path, Path]:
    if file_a and file_b:
        return Path(file_a), Path(file_b)
    reports_dir = Path(settings.reports_directory)
    files = sorted(
        [p for p in reports_dir.glob("benchmark_*.json") if p.name != "benchmark_latest.json"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    non_synthetic = [p for p in files if not _is_synthetic_benchmark_file(p)]
    selected = non_synthetic if len(non_synthetic) >= 2 else files
    if len(selected) < 2:
        raise ValueError("Need at least two benchmark files")
    return selected[1], selected[0]


def _summary_delta(summary_a: dict, summary_b: dict) -> dict:
    return {
        "successes": int(summary_b.get("successes", 0)) - int(summary_a.get("successes", 0)),
        "failures": int(summary_b.get("failures", 0)) - int(summary_a.get("failures", 0)),
        "avg_latency_s": round(float(summary_b.get("avg_latency_s", 0.0)) - float(summary_a.get("avg_latency_s", 0.0),), 3),
        "avg_reference_count": round(
            float(summary_b.get("avg_reference_count", 0.0)) - float(summary_a.get("avg_reference_count", 0.0)),
            2,
        ),
        "total_tokens": int(summary_b.get("total_tokens", 0)) - int(summary_a.get("total_tokens", 0)),
        "quality_gate_failures": int(summary_b.get("quality_gate_failures", 0))
        - int(summary_a.get("quality_gate_failures", 0)),
        "budget_failures": int(summary_b.get("budget_failures", 0)) - int(summary_a.get("budget_failures", 0)),
    }


def _iso_to_epoch(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp()


def _is_synthetic_benchmark_file(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    return int(summary.get("total_llm_calls", 0) or 0) == 0
