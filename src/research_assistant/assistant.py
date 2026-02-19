from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Dict
from uuid import uuid4

from .config import Settings
from .graph.workflow import create_research_workflow
from .services import CitationManager, DocumentParser, LLMService, ReportFormatter, SearchService
from .storage import SessionStore


class ResearchAssistant:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.workflow = create_research_workflow()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._session_lock = Lock()
        self._sessions: Dict[str, Dict] = {}
        self._store = SessionStore(self.settings.session_db_path)
        self.services = {
            "llm": LLMService(
                provider=self.settings.llm_provider,
                model=self.settings.llm_model,
                api_key=self.settings.openai_api_key,
                groq_api_key=self.settings.groq_api_key,
                xai_api_key=self.settings.xai_api_key,
                openrouter_api_key=self.settings.openrouter_api_key,
                ollama_base_url=self.settings.ollama_base_url,
                task_models={
                    "planning": self.settings.llm_model_planning,
                    "analysis": self.settings.llm_model_analysis,
                    "writing": self.settings.llm_model_writing,
                },
                fallback_enabled=self.settings.llm_route_fallback_enabled,
                fallback_provider=self.settings.llm_fallback_provider,
                fallback_model=self.settings.llm_fallback_model,
                second_fallback_provider=self.settings.llm_second_fallback_provider,
                second_fallback_model=self.settings.llm_second_fallback_model,
                retry_max_attempts=self.settings.llm_retry_max_attempts,
                retry_base_delay_s=self.settings.llm_retry_base_delay_s,
                retry_max_delay_s=self.settings.llm_retry_max_delay_s,
            ),
            "search": SearchService(
                provider=self.settings.search_provider,
                api_key=self.settings.tavily_api_key,
                serpapi_api_key=self.settings.serpapi_api_key,
            ),
            "parser": DocumentParser(),
            "citation": CitationManager(),
            "formatter": ReportFormatter(),
        }

    def research(self, query: str, progress_callback=None) -> Dict:
        # Citation state must be per run to avoid cross-session reference leakage.
        run_services = dict(self.services)
        run_services["citation"] = CitationManager()
        initial_state = {
            "query": query,
            "analyzed_query": {},
            "sub_questions": [],
            "current_question_idx": 0,
            "search_results": {},
            "raw_documents": {},
            "analyzed_content": [],
            "identified_gaps": [],
            "contradictions": [],
            "report_sections": {},
            "final_report": "",
            "metadata": {
                "warnings": [],
                "services": run_services,
                "settings": self.settings,
                "progress_callback": progress_callback,
            },
            "errors": [],
            "total_tokens_used": 0,
            "execution_time": 0.0,
            "iteration_count": 0,
        }
        return self.workflow.invoke(initial_state)

    def research_async(self, query: str) -> str:
        session_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._session_lock:
            self._sessions[session_id] = {
                "session_id": session_id,
                "query": query,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "error": None,
                "progress": {},
                "result": None,
                "future": None,
            }
            self._store.upsert(self._sessions[session_id])

        future = self._executor.submit(self._run_async_session, session_id, query)
        with self._session_lock:
            self._sessions[session_id]["future"] = future
        return session_id

    def get_progress(self, session_id: str) -> Dict:
        with self._session_lock:
            session = self._sessions.get(session_id)
        if session:
            return {
                "session_id": session_id,
                "status": session["status"],
                "updated_at": session["updated_at"],
                "error": session["error"],
                "progress": session.get("progress", {}),
            }
        stored = self._store.get(session_id)
        if not stored:
            return {"session_id": session_id, "status": "not_found"}
        return {
            "session_id": session_id,
            "status": stored["status"],
            "updated_at": stored["updated_at"],
            "error": stored["error"],
            "progress": {},
        }

    def get_result(self, session_id: str) -> Dict:
        with self._session_lock:
            session = self._sessions.get(session_id)
        if session:
            return {
                "session_id": session_id,
                "status": session["status"],
                "result": session["result"],
                "error": session["error"],
            }
        stored = self._store.get(session_id)
        if not stored:
            return {"session_id": session_id, "status": "not_found", "result": None}
        return {
            "session_id": session_id,
            "status": stored["status"],
            "result": stored["result"],
            "error": stored["error"],
        }

    def get_llm_metrics(self) -> Dict:
        llm = self.services["llm"]
        return llm.get_metrics()

    def reset_metrics_and_history(self) -> Dict:
        llm = self.services["llm"]
        llm.reset_metrics()
        cleared = self._store.clear_sessions()
        with self._session_lock:
            self._sessions = {}
        return {"cleared_sessions": cleared, "llm_metrics_cleared": True}

    def get_ops_metrics(self) -> Dict:
        llm_metrics = self.get_llm_metrics()
        llm_summary = []
        for row in llm_metrics.values():
            calls = row.get("calls", 0)
            successes = row.get("successes", 0)
            total_latency = float(row.get("total_latency_ms", 0.0))
            avg_latency = (total_latency / calls) if calls else 0.0
            success_rate = (successes / calls) if calls else 0.0
            total_tokens = int(row.get("total_tokens", 0))
            total_cost_usd = float(row.get("total_cost_usd", 0.0))
            llm_summary.append(
                {
                    "task": row.get("task", ""),
                    "model": row.get("model", ""),
                    "calls": calls,
                    "success_rate": round(success_rate, 4),
                    "avg_latency_ms": round(avg_latency, 2),
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(total_cost_usd, 8),
                    "fallback_calls": row.get("fallback_calls", 0),
                    "last_error": row.get("last_error", ""),
                }
            )

        with self._session_lock:
            active_sessions = sum(
                1
                for s in self._sessions.values()
                if s.get("status") in {"queued", "running"}
            )

        return {
            "active_sessions": active_sessions,
            "session_counts": self._store.count_by_status(),
            "recent_failures": self._store.recent_failures(limit=5),
            "llm_models": llm_summary,
        }

    def cancel_session(self, session_id: str) -> Dict:
        with self._session_lock:
            session = self._sessions.get(session_id)
            if not session:
                stored = self._store.get(session_id)
                if not stored:
                    return {"session_id": session_id, "status": "not_found"}
                return {"session_id": session_id, "status": stored["status"]}
            future = session.get("future")
            if future and future.cancel():
                self._update_session(session_id, status="cancelled")
                return {"session_id": session_id, "status": "cancelled"}
            if session["status"] in {"completed", "failed"}:
                return {"session_id": session_id, "status": session["status"]}
            return {"session_id": session_id, "status": session["status"]}

    def export_report(self, report_markdown: str, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_markdown, encoding="utf-8")

    def list_completed_reports(self, limit: int = 20) -> list[Dict]:
        return self._store.list_completed_reports(limit=limit)

    def _run_async_session(self, session_id: str, query: str) -> None:
        self._update_session(session_id, status="running")
        try:
            def on_progress(payload: Dict) -> None:
                self._update_session(session_id, progress=payload)

            result = self.research(query, progress_callback=on_progress)
            self._update_session(session_id, status="completed", result=result)
        except Exception as exc:  # pragma: no cover
            self._update_session(session_id, status="failed", error=str(exc))

    def _update_session(self, session_id: str, **fields) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._session_lock:
            current = self._sessions.get(session_id)
            if not current:
                stored = self._store.get(session_id)
                if not stored:
                    return
                current = stored
                self._sessions[session_id] = current
            if "progress" in fields and isinstance(fields["progress"], dict):
                merged = dict(current.get("progress", {}) or {})
                merged.update(fields["progress"])
                fields["progress"] = merged
            current.update(fields)
            current["updated_at"] = now
            self._store.upsert(current)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._store.close()
