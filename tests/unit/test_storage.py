from research_assistant.storage import SessionStore


def test_session_store_upsert_and_get(tmp_path):
    db_path = tmp_path / "sessions.db"
    store = SessionStore(str(db_path))

    payload = {
        "session_id": "s1",
        "query": "Impact of AI on education",
        "status": "completed",
        "created_at": "2026-02-18T00:00:00+00:00",
        "updated_at": "2026-02-18T00:00:01+00:00",
        "error": None,
        "result": {"final_report": "ok"},
    }
    store.upsert(payload)
    row = store.get("s1")
    assert row is not None
    assert row["status"] == "completed"
    assert row["result"]["final_report"] == "ok"
    store.close()

