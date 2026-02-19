import time

from fastapi.testclient import TestClient

from research_assistant.api import create_app
from research_assistant.assistant import ResearchAssistant
from research_assistant.config import Settings


def _client() -> TestClient:
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        max_research_iterations=5,
        min_relevance_score=0.8,
    )
    assistant = ResearchAssistant(settings=settings)
    return TestClient(create_app(assistant=assistant))


def test_submit_research_endpoint():
    client = _client()
    response = client.post("/api/v1/research", json={"query": "Impact of AI on education"})
    assert response.status_code == 202
    body = response.json()
    assert "session_id" in body
    assert body["status"] == "queued"


def test_status_and_result_endpoints():
    client = _client()
    create = client.post("/api/v1/research", json={"query": "Benefits of renewable energy"})
    session_id = create.json()["session_id"]

    pending = client.get(f"/api/v1/research/{session_id}/result")
    assert pending.status_code in {200, 409}

    deadline = time.time() + 5
    status = "queued"
    while time.time() < deadline:
        response = client.get(f"/api/v1/research/{session_id}/status")
        assert response.status_code == 200
        body = response.json()
        status = body["status"]
        assert "progress" in body
        if status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.05)

    assert status == "completed"
    result = client.get(f"/api/v1/research/{session_id}/result")
    assert result.status_code == 200
    payload = result.json()
    assert payload["result"]["final_report"]


def test_not_found_endpoints():
    client = _client()
    missing = "missing-session"
    status = client.get(f"/api/v1/research/{missing}/status")
    assert status.status_code == 404
    result = client.get(f"/api/v1/research/{missing}/result")
    assert result.status_code == 404
    cancel = client.post(f"/api/v1/research/{missing}/cancel")
    assert cancel.status_code == 404


def test_api_recovery_across_app_instances(tmp_path):
    db_path = tmp_path / "sessions.db"
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        max_research_iterations=5,
        min_relevance_score=0.8,
        session_db_path=str(db_path),
    )

    assistant_a = ResearchAssistant(settings=settings)
    client_a = TestClient(create_app(assistant=assistant_a))
    create = client_a.post("/api/v1/research", json={"query": "AI in healthcare adoption trends"})
    session_id = create.json()["session_id"]

    deadline = time.time() + 5
    status = "queued"
    while time.time() < deadline:
        resp = client_a.get(f"/api/v1/research/{session_id}/status")
        status = resp.json()["status"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert status == "completed"
    assistant_a.close()

    assistant_b = ResearchAssistant(settings=settings)
    client_b = TestClient(create_app(assistant=assistant_b))
    restored_status = client_b.get(f"/api/v1/research/{session_id}/status")
    assert restored_status.status_code == 200
    assert restored_status.json()["status"] == "completed"

    restored_result = client_b.get(f"/api/v1/research/{session_id}/result")
    assert restored_result.status_code == 200
    assert restored_result.json()["result"]["final_report"]
    assistant_b.close()


def test_llm_metrics_endpoint_with_routed_models(monkeypatch):
    class _FakeResponse:
        def __init__(self, content: str):
            self._content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": self._content}}]}

    def fake_post(url, headers=None, json=None, timeout=0):
        prompt = json["messages"][0]["content"]
        if "Classify the research query" in prompt:
            content = '{"query_type":"exploratory","scope":"broad"}'
        elif "Break this query into" in prompt:
            content = (
                '{"sub_questions":['
                '{"question":"Q1 impact","priority":1,"answered":false,"parent_question":null},'
                '{"question":"Q2 risks","priority":2,"answered":false,"parent_question":null},'
                '{"question":"Q3 outlook","priority":3,"answered":false,"parent_question":null}'
                ']}'
            )
        elif "Extract key factual claims" in prompt:
            subq = prompt.split("Sub-question: ", 1)[1].split("\n", 1)[0]
            content = (
                '{"key_information":["'
                + subq
                + ' evidence"],'
                '"citations":["https://example.com/source"],'
                '"contradictions":[]}'
            )
        elif "Create a short executive summary" in prompt:
            content = "Summary."
        else:
            content = "Section body."
        return _FakeResponse(content)

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)

    settings = Settings(
        llm_provider="openrouter",
        llm_model="openai/gpt-oss-120b:free",
        llm_model_planning="openai/gpt-oss-20b:free",
        llm_model_analysis="deepseek/deepseek-r1-0528:free",
        llm_model_writing="openai/gpt-oss-120b:free",
        openrouter_api_key="or-test-key",
        search_provider="mock",
        max_research_iterations=3,
        min_relevance_score=0.8,
    )
    assistant = ResearchAssistant(settings=settings)
    client = TestClient(create_app(assistant=assistant))

    response = client.post("/api/v1/research", json={"query": "Impact of AI on education"})
    assert response.status_code == 202
    session_id = response.json()["session_id"]

    deadline = time.time() + 5
    status = "queued"
    while time.time() < deadline:
        status_resp = client.get(f"/api/v1/research/{session_id}/status")
        status = status_resp.json()["status"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert status == "completed"

    metrics = client.get("/api/v1/metrics/llm")
    assert metrics.status_code == 200
    payload = metrics.json()["metrics"]
    assert any("planning:" in key for key in payload)
    assert any("analysis:" in key for key in payload)
    assert any("writing:" in key for key in payload)


def test_ops_metrics_endpoint_success_case(monkeypatch):
    class _FakeResponse:
        def __init__(self, content: str):
            self._content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": self._content}}]}

    def fake_post(url, headers=None, json=None, timeout=0):
        prompt = json["messages"][0]["content"]
        if "Classify the research query" in prompt:
            content = '{"query_type":"exploratory","scope":"broad"}'
        elif "Break this query into" in prompt:
            content = (
                '{"sub_questions":['
                '{"question":"Q1 impact","priority":1,"answered":false,"parent_question":null},'
                '{"question":"Q2 risks","priority":2,"answered":false,"parent_question":null},'
                '{"question":"Q3 outlook","priority":3,"answered":false,"parent_question":null}'
                ']}'
            )
        elif "Extract key factual claims" in prompt:
            content = '{"key_information":["evidence"],"citations":["https://example.com"],"contradictions":[]}'
        elif "Create a short executive summary" in prompt:
            content = "Summary."
        else:
            content = "Section body."
        return _FakeResponse(content)

    monkeypatch.setattr("research_assistant.services.llm.requests.post", fake_post)
    settings = Settings(
        llm_provider="openrouter",
        llm_model="openai/gpt-oss-120b:free",
        openrouter_api_key="or-test-key",
        search_provider="mock",
    )
    assistant = ResearchAssistant(settings=settings)
    client = TestClient(create_app(assistant=assistant))
    created = client.post("/api/v1/research", json={"query": "Impact of AI on education"})
    sid = created.json()["session_id"]

    deadline = time.time() + 5
    while time.time() < deadline:
        s = client.get(f"/api/v1/research/{sid}/status").json()["status"]
        if s in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert s == "completed"

    ops = client.get("/api/v1/metrics/ops")
    assert ops.status_code == 200
    payload = ops.json()
    assert "active_sessions" in payload
    assert "session_counts" in payload
    assert "recent_failures" in payload
    assert "llm_models" in payload
    assert payload["session_counts"].get("completed", 0) >= 1


def test_ops_metrics_endpoint_failure_capture():
    settings = Settings(
        llm_provider="openrouter",
        llm_model="openai/gpt-oss-120b:free",
        openrouter_api_key="",  # Force failure
        llm_route_fallback_enabled=False,
        search_provider="mock",
        llm_retry_max_attempts=1,
    )
    assistant = ResearchAssistant(settings=settings)
    client = TestClient(create_app(assistant=assistant))
    created = client.post("/api/v1/research", json={"query": "Will fail due to missing key"})
    sid = created.json()["session_id"]

    deadline = time.time() + 5
    status = "queued"
    while time.time() < deadline:
        status = client.get(f"/api/v1/research/{sid}/status").json()["status"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert status == "failed"

    ops = client.get("/api/v1/metrics/ops")
    assert ops.status_code == 200
    payload = ops.json()
    assert payload["session_counts"].get("failed", 0) >= 1
    assert any(item["session_id"] == sid for item in payload["recent_failures"])


def test_api_auth_token_required():
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        api_auth_token="secret-token",
    )
    assistant = ResearchAssistant(settings=settings)
    client = TestClient(create_app(assistant=assistant))

    unauthorized = client.get("/api/v1/metrics/llm")
    assert unauthorized.status_code == 401

    authorized = client.get("/api/v1/metrics/llm", headers={"x-api-key": "secret-token"})
    assert authorized.status_code == 200


def test_api_rate_limit_per_minute():
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        api_rate_limit_per_minute=2,
    )
    assistant = ResearchAssistant(settings=settings)
    client = TestClient(create_app(assistant=assistant))

    first = client.get("/api/v1/metrics/llm")
    second = client.get("/api/v1/metrics/llm")
    third = client.get("/api/v1/metrics/llm")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429


def test_dashboard_route_serves_html():
    client = _client()
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Research Assistant Dashboard" in response.text


def test_reports_and_benchmark_api_endpoints(tmp_path):
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        reports_directory=str(tmp_path / "reports"),
        session_db_path=str(tmp_path / "sessions.db"),
    )
    assistant = ResearchAssistant(settings=settings)
    client = TestClient(create_app(assistant=assistant))

    create = client.post("/api/v1/research", json={"query": "Impact of AI on education"})
    sid = create.json()["session_id"]
    deadline = time.time() + 5
    status = "queued"
    while time.time() < deadline:
        status = client.get(f"/api/v1/research/{sid}/status").json()["status"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert status == "completed"

    reports = client.get("/api/v1/reports?limit=5&kind=session")
    assert reports.status_code == 200
    entries = reports.json()["entries"]
    assert len(entries) >= 1
    assert entries[0]["kind"] == "session"

    content = client.get(f"/api/v1/reports/content?ref=session:{sid}")
    assert content.status_code == 200
    assert "markdown" in content.json()

    # seed two benchmark files so compare works
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "benchmark_20260101_000000.json").write_text(
        '{"summary":{"successes":1,"failures":1,"avg_latency_s":10.0,"avg_reference_count":2.0,"total_tokens":100}}',
        encoding="utf-8",
    )
    (reports_dir / "benchmark_20260101_000001.json").write_text(
        '{"summary":{"successes":2,"failures":0,"avg_latency_s":8.0,"avg_reference_count":3.0,"total_tokens":120}}',
        encoding="utf-8",
    )
    history = client.get("/api/v1/benchmarks/history?limit=5")
    assert history.status_code == 200
    assert len(history.json()["entries"]) >= 2

    compare = client.get(
        "/api/v1/benchmarks/compare",
        params={
            "file_a": str(reports_dir / "benchmark_20260101_000000.json"),
            "file_b": str(reports_dir / "benchmark_20260101_000001.json"),
        },
    )
    assert compare.status_code == 200
    delta = compare.json()["delta_b_minus_a"]
    assert delta["successes"] == 1
    assert delta["failures"] == -1


def test_benchmark_compare_api_defaults_skip_synthetic(tmp_path):
    settings = Settings(
        llm_provider="mock",
        search_provider="mock",
        reports_directory=str(tmp_path / "reports"),
        session_db_path=str(tmp_path / "sessions.db"),
    )
    assistant = ResearchAssistant(settings=settings)
    client = TestClient(create_app(assistant=assistant))

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "benchmark_20260101_000000.json").write_text(
        '{"summary":{"successes":1,"failures":1,"total_tokens":100,"total_llm_calls":10}}',
        encoding="utf-8",
    )
    (reports_dir / "benchmark_20260101_000001.json").write_text(
        '{"summary":{"successes":2,"failures":0,"total_tokens":120,"total_llm_calls":12}}',
        encoding="utf-8",
    )
    (reports_dir / "benchmark_20260101_000002.json").write_text(
        '{"summary":{"successes":3,"failures":0,"total_tokens":0,"total_llm_calls":0}}',
        encoding="utf-8",
    )

    compare = client.get("/api/v1/benchmarks/compare")
    assert compare.status_code == 200
    payload = compare.json()
    assert not payload["file_a"].endswith("benchmark_20260101_000002.json")
    assert not payload["file_b"].endswith("benchmark_20260101_000002.json")
    compared = {payload["file_a"].split("\\")[-1], payload["file_b"].split("\\")[-1]}
    assert compared == {"benchmark_20260101_000000.json", "benchmark_20260101_000001.json"}
    delta = payload["delta_b_minus_a"]
    assert abs(delta["successes"]) == 1
    assert abs(delta["failures"]) == 1
