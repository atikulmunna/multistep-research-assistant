import json

from typer.testing import CliRunner

from research_assistant.main import app


def test_benchmark_command_json_output(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    monkeypatch.setenv("REPORTS_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))

    runner = CliRunner()
    result = runner.invoke(app, ["benchmark", "--num-queries", "2", "--json", "--reset-first"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "summary" in payload
    assert payload["summary"]["queries_run"] == 2
    assert "quality_gate_failures" in payload["summary"]
    assert "history_file" in payload
    assert "runs" in payload
    assert len(payload["runs"]) == 2
    assert (tmp_path / "benchmark_latest.json").exists()


def test_benchmark_queries_file(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))

    queries_file = tmp_path / "queries.txt"
    queries_file.write_text("Q one\nQ two\nQ three\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["benchmark", "--queries-file", str(queries_file), "--num-queries", "2", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["runs"][0]["query"] == "Q one"
    assert payload["runs"][1]["query"] == "Q two"


def test_benchmark_continues_on_query_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))

    calls = {"n": 0}

    def fake_research(self, query):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated failure")
        return {"final_report": "# ok\n\n## References\n[1] https://example.com"}

    monkeypatch.setattr("research_assistant.main.ResearchAssistant.research", fake_research)

    runner = CliRunner()
    result = runner.invoke(app, ["benchmark", "--num-queries", "2", "--json", "--reset-first"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["summary"]["queries_run"] == 2
    assert payload["summary"]["successes"] == 1
    assert payload["summary"]["failures"] == 1
    assert payload["runs"][0]["success"] is False
    assert "simulated failure" in payload["runs"][0]["error"]
    assert payload["runs"][0]["error_type"] == "runtimeerror"


def test_benchmark_quality_gate_enforced(monkeypatch, tmp_path):
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setenv("QUALITY_GATE_ENFORCE", "true")

    def fake_research(self, query):
        return {
            "final_report": "# report\n\n## References\n[1] https://example.com",
            "metadata": {
                "quality": {
                    "passed": False,
                    "checks_failed": ["reference_count_below_min"],
                }
            },
        }

    monkeypatch.setattr("research_assistant.main.ResearchAssistant.research", fake_research)

    runner = CliRunner()
    result = runner.invoke(app, ["benchmark", "--num-queries", "1", "--json", "--reset-first"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["summary"]["quality_gate_failures"] == 1
    assert payload["runs"][0]["success"] is False
    assert payload["runs"][0]["error_type"] == "quality_gate_failed"
    assert payload["summary"]["budget_failures"] == 0


def test_benchmark_budget_exceeded_on_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setenv("MAX_TOTAL_TOKENS_PER_QUERY", "10")

    def fake_research(self, query):
        return {"final_report": "# report\n\n## References\n[1] https://example.com"}

    token_calls = {"n": 0}

    def fake_metrics(self):
        token_calls["n"] += 1
        if token_calls["n"] <= 1:
            return {}
        return {"x": {"total_tokens": 25}}

    monkeypatch.setattr("research_assistant.main.ResearchAssistant.research", fake_research)
    monkeypatch.setattr("research_assistant.main.ResearchAssistant.get_llm_metrics", fake_metrics)

    runner = CliRunner()
    result = runner.invoke(app, ["benchmark", "--num-queries", "1", "--json", "--reset-first"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["runs"][0]["success"] is False
    assert payload["runs"][0]["error_type"] == "budget_exceeded"
    assert payload["runs"][0]["query_total_tokens"] >= 25
    assert payload["summary"]["budget_failures"] == 1


def test_benchmark_budget_exceeded_on_duration(monkeypatch, tmp_path):
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setenv("MAX_SECONDS_PER_QUERY", "0.001")

    def fake_research(self, query):
        import time as _time

        _time.sleep(0.01)
        return {"final_report": "# report\n\n## References\n[1] https://example.com"}

    monkeypatch.setattr("research_assistant.main.ResearchAssistant.research", fake_research)

    runner = CliRunner()
    result = runner.invoke(app, ["benchmark", "--num-queries", "1", "--json", "--reset-first"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["runs"][0]["success"] is False
    assert payload["runs"][0]["error_type"] == "budget_exceeded"
    assert payload["summary"]["budget_failures"] == 1


def test_benchmark_adaptive_depth_reruns_on_quality_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setenv("ADAPTIVE_DEPTH_ENABLED", "true")
    monkeypatch.setenv("ADAPTIVE_MAX_PASSES", "1")
    monkeypatch.setenv("QUALITY_GATE_ENFORCE", "true")

    calls = {"n": 0}
    observed = []

    def fake_research(self, query):
        calls["n"] += 1
        observed.append((self.settings.max_sub_questions, self.settings.max_research_iterations))
        if calls["n"] == 1:
            return {
                "final_report": "# report\n\n## References\n[1] https://example.com",
                "metadata": {"quality": {"passed": False, "checks_failed": ["reference_count_below_min"]}},
            }
        return {
            "final_report": "# report\n\n## References\n[1] https://example.com\n[2] https://example.org",
            "metadata": {"quality": {"passed": True, "checks_failed": []}},
        }

    monkeypatch.setattr("research_assistant.main.ResearchAssistant.research", fake_research)

    runner = CliRunner()
    result = runner.invoke(app, ["benchmark", "--num-queries", "1", "--json", "--reset-first"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["summary"]["adaptive_attempts"] == 1
    assert payload["summary"]["adaptive_applied"] == 1
    assert payload["runs"][0]["adaptive_applied"] is True
    assert payload["runs"][0]["success"] is True
    assert calls["n"] == 2
    assert observed[1][0] > observed[0][0]
    assert observed[1][1] > observed[0][1]


def test_benchmark_adaptive_disabled_no_rerun(monkeypatch, tmp_path):
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setenv("ADAPTIVE_DEPTH_ENABLED", "false")
    monkeypatch.setenv("QUALITY_GATE_ENFORCE", "true")

    calls = {"n": 0}

    def fake_research(self, query):
        calls["n"] += 1
        return {
            "final_report": "# report\n\n## References\n[1] https://example.com",
            "metadata": {"quality": {"passed": False, "checks_failed": ["reference_count_below_min"]}},
        }

    monkeypatch.setattr("research_assistant.main.ResearchAssistant.research", fake_research)

    runner = CliRunner()
    result = runner.invoke(app, ["benchmark", "--num-queries", "1", "--json", "--reset-first"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["summary"]["adaptive_attempts"] == 0
    assert payload["summary"]["adaptive_applied"] == 0
    assert payload["runs"][0]["adaptive_applied"] is False
    assert payload["runs"][0]["success"] is False
    assert calls["n"] == 1


def test_benchmark_pause_between_queries(monkeypatch, tmp_path):
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setattr(
        "research_assistant.main.ResearchAssistant.research",
        lambda self, query: {"final_report": "# ok\n\n## References\n[1] https://example.com"},
    )

    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("research_assistant.main.time.sleep", fake_sleep)

    runner = CliRunner()
    result = runner.invoke(app, ["benchmark", "--num-queries", "3", "--pause-seconds", "0.5", "--json"])
    assert result.exit_code == 0
    assert sleeps == [0.5, 0.5]


def test_benchmark_history_command(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    monkeypatch.setenv("REPORTS_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))

    runner = CliRunner()
    run_result = runner.invoke(app, ["benchmark", "--num-queries", "1", "--json"])
    assert run_result.exit_code == 0

    history_result = runner.invoke(app, ["benchmark-history", "--json", "--limit", "5"])
    assert history_result.exit_code == 0
    payload = json.loads(history_result.stdout)
    assert payload["count"] >= 1
    assert payload["entries"][0]["queries_run"] == 1


def test_benchmark_compare_command(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORTS_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))

    first = {
        "summary": {
            "queries_run": 2,
            "successes": 1,
            "failures": 1,
            "quality_gate_failures": 1,
            "avg_latency_s": 10.0,
            "avg_reference_count": 2.0,
            "total_tokens": 100,
        }
    }
    second = {
        "summary": {
            "queries_run": 2,
            "successes": 2,
            "failures": 0,
            "quality_gate_failures": 0,
            "avg_latency_s": 8.0,
            "avg_reference_count": 3.0,
            "total_tokens": 120,
        }
    }
    f1 = tmp_path / "benchmark_20260101_000000.json"
    f2 = tmp_path / "benchmark_20260101_000001.json"
    f1.write_text(json.dumps(first), encoding="utf-8")
    f2.write_text(json.dumps(second), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark-compare",
            "--file-a",
            str(f1),
            "--file-b",
            str(f2),
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    delta = payload["delta_b_minus_a"]
    assert delta["successes"] == 1
    assert delta["failures"] == -1
    assert delta["avg_latency_s"] == -2.0
    assert delta["avg_reference_count"] == 1.0
    assert delta["total_tokens"] == 20


def test_benchmark_compare_defaults_skip_synthetic(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORTS_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("SESSION_DB_PATH", str(tmp_path / "sessions.db"))

    real_a = {
        "summary": {
            "queries_run": 2,
            "successes": 1,
            "failures": 1,
            "quality_gate_failures": 1,
            "avg_latency_s": 10.0,
            "avg_reference_count": 2.0,
            "total_tokens": 100,
            "total_llm_calls": 10,
        }
    }
    real_b = {
        "summary": {
            "queries_run": 2,
            "successes": 2,
            "failures": 0,
            "quality_gate_failures": 0,
            "avg_latency_s": 8.0,
            "avg_reference_count": 3.0,
            "total_tokens": 120,
            "total_llm_calls": 12,
        }
    }
    synthetic = {
        "summary": {
            "queries_run": 3,
            "successes": 3,
            "failures": 0,
            "quality_gate_failures": 0,
            "avg_latency_s": 0.0,
            "avg_reference_count": 1.0,
            "total_tokens": 0,
            "total_llm_calls": 0,
        }
    }
    f1 = tmp_path / "benchmark_20260101_000000.json"
    f2 = tmp_path / "benchmark_20260101_000001.json"
    f3 = tmp_path / "benchmark_20260101_000002.json"
    f1.write_text(json.dumps(real_a), encoding="utf-8")
    f2.write_text(json.dumps(real_b), encoding="utf-8")
    f3.write_text(json.dumps(synthetic), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["benchmark-compare", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert not payload["file_a"].endswith("benchmark_20260101_000002.json")
    assert not payload["file_b"].endswith("benchmark_20260101_000002.json")
    compared = {payload["file_a"].split("\\")[-1], payload["file_b"].split("\\")[-1]}
    assert compared == {"benchmark_20260101_000000.json", "benchmark_20260101_000001.json"}
    delta = payload["delta_b_minus_a"]
    assert abs(delta["successes"]) == 1
    assert abs(delta["failures"]) == 1
