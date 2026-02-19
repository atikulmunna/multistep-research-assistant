import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


class SessionStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def upsert(self, session: Dict[str, Any]) -> None:
        result_json = None
        if session.get("result") is not None:
            result_json = json.dumps(session["result"], default=str)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (
                    session_id, query, status, created_at, updated_at, error, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    query = excluded.query,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    error = excluded.error,
                    result_json = excluded.result_json
                """,
                (
                    session["session_id"],
                    session.get("query", ""),
                    session.get("status", "queued"),
                    session.get("created_at"),
                    session.get("updated_at"),
                    session.get("error"),
                    result_json,
                ),
            )
            self._conn.commit()

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT session_id, query, status, created_at, updated_at, error, result_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        result = json.loads(row["result_json"]) if row["result_json"] else None
        return {
            "session_id": row["session_id"],
            "query": row["query"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "error": row["error"],
            "result": result,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def count_by_status(self) -> Dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS cnt FROM sessions GROUP BY status"
            ).fetchall()
        return {row["status"]: int(row["cnt"]) for row in rows}

    def recent_failures(self, limit: int = 5) -> list[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT session_id, query, status, updated_at, error
                FROM sessions
                WHERE status = 'failed'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "query": row["query"],
                "status": row["status"],
                "updated_at": row["updated_at"],
                "error": row["error"],
            }
            for row in rows
        ]

    def clear_sessions(self) -> int:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM sessions")
            self._conn.commit()
            return int(cursor.rowcount)

    def list_completed_reports(self, limit: int = 20) -> list[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT session_id, query, status, created_at, updated_at, result_json
                FROM sessions
                WHERE status = 'completed' AND result_json IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out = []
        for row in rows:
            result = None
            if row["result_json"]:
                try:
                    result = json.loads(row["result_json"])
                except json.JSONDecodeError:
                    result = None
            report = ""
            if isinstance(result, dict):
                report = str(result.get("final_report", "") or "")
            out.append(
                {
                    "ref": f"session:{row['session_id']}",
                    "kind": "session",
                    "session_id": row["session_id"],
                    "query": row["query"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "report_chars": len(report),
                    "has_report": bool(report),
                }
            )
        return out

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    result_json TEXT
                )
                """
            )
            self._conn.commit()
