from pathlib import Path
import json
import time
from datetime import datetime, timezone
import html
import webbrowser

import typer

from .assistant import ResearchAssistant
from .config import Settings

app = typer.Typer(add_completion=False, no_args_is_help=True)
reports_app = typer.Typer(help="Browse and export generated reports")
app.add_typer(reports_app, name="reports")


@app.command()
def run(
    query: str = typer.Argument(..., help="Research query"),
    output: str = typer.Option("", "--output", "-o", help="Optional output markdown file"),
) -> None:
    settings = Settings()
    assistant = ResearchAssistant(settings=settings)
    try:
        result = assistant.research(query)
        report = result["final_report"]

        if output:
            assistant.export_report(report, output)
            typer.echo(f"Report written to {output}")
        else:
            typer.echo(report)
    finally:
        assistant.close()


@app.command()
def sample(
    output: str = typer.Option("reports/sample_report.md", "--output", "-o", help="Output markdown file"),
) -> None:
    settings = Settings()
    assistant = ResearchAssistant(settings=settings)
    try:
        result = assistant.research("Impact of AI on education")
        assistant.export_report(result["final_report"], output)
        typer.echo(f"Sample report written to {Path(output)}")
    finally:
        assistant.close()


@app.command()
def watch(
    session_id: str = typer.Argument(..., help="Session id to watch"),
    interval: float = typer.Option(0.5, "--interval", min=0.1, help="Polling interval in seconds"),
    timeout: float = typer.Option(120.0, "--timeout", min=1.0, help="Max watch duration in seconds"),
) -> None:
    settings = Settings()
    assistant = ResearchAssistant(settings=settings)
    try:
        deadline = time.perf_counter() + timeout
        while True:
            progress = assistant.get_progress(session_id)
            status = progress.get("status", "unknown")
            payload = progress.get("progress", {}) or {}
            stage = payload.get("stage", "")
            sub_done = payload.get("sub_questions_done")
            sub_total = payload.get("sub_questions_total")
            iteration = payload.get("iteration")
            line = f"status={status}"
            if stage:
                line += f" stage={stage}"
            if isinstance(sub_done, int) and isinstance(sub_total, int):
                line += f" sub_questions={sub_done}/{sub_total}"
            if isinstance(iteration, int):
                line += f" iteration={iteration}"
            typer.echo(line)
            if status in {"completed", "failed", "cancelled", "not_found"}:
                break
            if time.perf_counter() >= deadline:
                typer.echo("watch timeout reached")
                break
            time.sleep(interval)
    finally:
        assistant.close()


@app.command()
def metrics(
    as_json: bool = typer.Option(False, "--json", help="Print raw metrics as JSON"),
    reset: bool = typer.Option(False, "--reset", help="Clear LLM metrics and persisted session history"),
    yes: bool = typer.Option(False, "--yes", help="Skip reset confirmation prompt"),
) -> None:
    settings = Settings()
    assistant = ResearchAssistant(settings=settings)
    try:
        if reset:
            if not yes:
                confirmed = typer.confirm("Reset all metrics and session history?")
                if not confirmed:
                    typer.echo("Reset cancelled.")
                    return
            outcome = assistant.reset_metrics_and_history()
            if as_json:
                typer.echo(json.dumps({"reset": outcome}, indent=2))
            else:
                typer.echo(f"Reset complete: cleared_sessions={outcome['cleared_sessions']}")
            return

        llm_metrics = assistant.get_llm_metrics()
        ops_metrics = assistant.get_ops_metrics()
        payload = {"llm": llm_metrics, "ops": ops_metrics}
        if as_json:
            typer.echo(json.dumps(payload, indent=2))
            return

        typer.echo("LLM Metrics")
        if not llm_metrics:
            typer.echo("- no llm metrics collected yet")
        for key, value in llm_metrics.items():
            calls = value.get("calls", 0)
            successes = value.get("successes", 0)
            fallback_calls = value.get("fallback_calls", 0)
            avg_ms = (value.get("total_latency_ms", 0.0) / calls) if calls else 0.0
            typer.echo(
                f"- {key}: calls={calls} successes={successes} "
                f"fallback_calls={fallback_calls} avg_latency_ms={avg_ms:.2f}"
            )

        typer.echo("\nOps Metrics")
        typer.echo(f"- active_sessions={ops_metrics.get('active_sessions', 0)}")
        typer.echo(f"- session_counts={ops_metrics.get('session_counts', {})}")
        failures = ops_metrics.get("recent_failures", [])
        typer.echo(f"- recent_failures={len(failures)}")
    finally:
        assistant.close()


@app.command()
def benchmark(
    num_queries: int = typer.Option(3, "--num-queries", min=1, help="How many queries to run"),
    queries_file: str = typer.Option("", "--queries-file", help="Optional file with one query per line"),
    as_json: bool = typer.Option(False, "--json", help="Print benchmark output as JSON"),
    reset_first: bool = typer.Option(False, "--reset-first", help="Clear stored history and in-memory metrics before run"),
    pause_seconds: float = typer.Option(0.0, "--pause-seconds", min=0.0, help="Pause between queries to reduce rate-limit pressure"),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Stop benchmark on first query failure"),
) -> None:
    settings = Settings()
    assistant = ResearchAssistant(settings=settings)
    try:
        if reset_first:
            assistant.reset_metrics_and_history()

        queries = _load_benchmark_queries(queries_file)
        selected = queries[:num_queries]
        runs = []
        for idx, query in enumerate(selected):
            before_tokens = _total_tokens_from_metrics(assistant.get_llm_metrics())
            started = time.perf_counter()
            try:
                result = assistant.research(query)
                duration_s = time.perf_counter() - started
                after_tokens = _total_tokens_from_metrics(assistant.get_llm_metrics())
                query_total_tokens = max(0, after_tokens - before_tokens)
                report = result.get("final_report", "")
                quality = _extract_quality(result)
                quality_passed = bool(quality.get("passed", True))
                adaptive_attempted = False
                adaptive_applied = False

                if _should_try_adaptive(settings=settings, quality_passed=quality_passed):
                    adaptive_attempted = True
                    if _within_budget_headroom(
                        duration_s=duration_s,
                        query_total_tokens=query_total_tokens,
                        max_seconds=float(getattr(settings, "max_seconds_per_query", 0.0)),
                        max_tokens=int(getattr(settings, "max_total_tokens_per_query", 0)),
                    ):
                        result, duration_s, query_total_tokens = _run_adaptive_passes(
                            assistant=assistant,
                            query=query,
                            base_result=result,
                            base_duration_s=duration_s,
                            before_tokens=before_tokens,
                            settings=settings,
                        )
                        report = result.get("final_report", "")
                        quality = _extract_quality(result)
                        quality_passed = bool(quality.get("passed", True))
                        adaptive_applied = True

                enforce_quality = bool(getattr(settings, "quality_gate_enforce", False))
                success = bool(report) and (quality_passed or not enforce_quality)
                err_msg = ""
                err_type = ""
                if not quality_passed and enforce_quality:
                    checks = ",".join(quality.get("checks_failed", []))
                    err_msg = f"quality_gate_failed:{checks}"
                    err_type = "quality_gate_failed"
                budget_reasons = []
                max_tokens = int(getattr(settings, "max_total_tokens_per_query", 0))
                max_seconds = float(getattr(settings, "max_seconds_per_query", 0.0))
                if max_tokens > 0 and query_total_tokens > max_tokens:
                    budget_reasons.append(f"tokens={query_total_tokens}>{max_tokens}")
                if max_seconds > 0 and duration_s > max_seconds:
                    budget_reasons.append(f"seconds={round(duration_s, 3)}>{max_seconds}")
                if budget_reasons:
                    success = False
                    err_msg = f"budget_exceeded:{','.join(budget_reasons)}"
                    err_type = "budget_exceeded"
                runs.append(
                    {
                        "query": query,
                        "duration_s": round(duration_s, 3),
                        "report_chars": len(report),
                        "reference_count": _count_references(report),
                        "query_total_tokens": query_total_tokens,
                        "adaptive_attempted": adaptive_attempted,
                        "adaptive_applied": adaptive_applied,
                        "success": success,
                        "error": err_msg,
                        "error_type": err_type,
                        "quality_passed": quality_passed,
                        "quality_checks_failed": quality.get("checks_failed", []),
                    }
                )
            except Exception as exc:
                duration_s = time.perf_counter() - started
                after_tokens = _total_tokens_from_metrics(assistant.get_llm_metrics())
                query_total_tokens = max(0, after_tokens - before_tokens)
                error = f"{type(exc).__name__}: {exc}"
                runs.append(
                    {
                        "query": query,
                        "duration_s": round(duration_s, 3),
                        "report_chars": 0,
                        "reference_count": 0,
                        "query_total_tokens": query_total_tokens,
                        "adaptive_attempted": False,
                        "adaptive_applied": False,
                        "success": False,
                        "error": error,
                        "error_type": _classify_error_type(error),
                        "quality_passed": False,
                        "quality_checks_failed": [],
                    }
                )
                if fail_fast:
                    raise

            if pause_seconds > 0 and idx < len(selected) - 1:
                time.sleep(pause_seconds)

        llm_metrics = assistant.get_llm_metrics()
        ops_metrics = assistant.get_ops_metrics()
        durations = [r["duration_s"] for r in runs]
        ref_counts = [r["reference_count"] for r in runs]
        report_sizes = [r["report_chars"] for r in runs]

        total_calls = sum(int(v.get("calls", 0)) for v in llm_metrics.values())
        total_tokens = sum(int(v.get("total_tokens", 0)) for v in llm_metrics.values())
        total_cost_usd = sum(float(v.get("total_cost_usd", 0.0)) for v in llm_metrics.values())

        summary = {
            "queries_run": len(runs),
            "successes": sum(1 for r in runs if r["success"]),
            "failures": sum(1 for r in runs if not r["success"]),
            "quality_gate_failures": sum(1 for r in runs if r.get("error_type") == "quality_gate_failed"),
            "budget_failures": sum(1 for r in runs if r.get("error_type") == "budget_exceeded"),
            "adaptive_attempts": sum(1 for r in runs if r.get("adaptive_attempted")),
            "adaptive_applied": sum(1 for r in runs if r.get("adaptive_applied")),
            "avg_latency_s": round(sum(durations) / len(durations), 3) if durations else 0.0,
            "max_latency_s": round(max(durations), 3) if durations else 0.0,
            "avg_report_chars": int(sum(report_sizes) / len(report_sizes)) if report_sizes else 0,
            "avg_reference_count": round(sum(ref_counts) / len(ref_counts), 2) if ref_counts else 0.0,
            "total_llm_calls": total_calls,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost_usd, 8),
        }
        payload = {"summary": summary, "runs": runs, "llm_metrics": llm_metrics, "ops_metrics": ops_metrics}
        history_path = _save_benchmark_payload(payload=payload, settings=settings)
        payload["history_file"] = str(history_path)

        if as_json:
            typer.echo(json.dumps(payload, indent=2))
            return

        typer.echo("Benchmark Summary")
        typer.echo(
            f"- queries_run={summary['queries_run']} "
            f"successes={summary['successes']} failures={summary['failures']} "
            f"quality_gate_failures={summary['quality_gate_failures']} "
            f"budget_failures={summary['budget_failures']} "
            f"adaptive_attempts={summary['adaptive_attempts']} adaptive_applied={summary['adaptive_applied']}"
        )
        typer.echo(f"- avg_latency_s={summary['avg_latency_s']} max_latency_s={summary['max_latency_s']}")
        typer.echo(f"- avg_report_chars={summary['avg_report_chars']} avg_reference_count={summary['avg_reference_count']}")
        typer.echo(f"- total_llm_calls={summary['total_llm_calls']} total_tokens={summary['total_tokens']} total_cost_usd={summary['total_cost_usd']}")
        typer.echo("\nPer-run")
        for r in runs:
            typer.echo(
                f"- query={r['query']} duration_s={r['duration_s']} "
                f"refs={r['reference_count']} tokens={r.get('query_total_tokens',0)} report_chars={r['report_chars']} "
                f"adaptive_applied={r.get('adaptive_applied',False)} "
                f"success={r['success']} error_type={r.get('error_type','')} error={r['error']}"
            )
        typer.echo(f"\nSaved benchmark: {history_path}")
    finally:
        assistant.close()


@app.command("benchmark-history")
def benchmark_history(
    as_json: bool = typer.Option(False, "--json", help="Print benchmark history as JSON"),
    limit: int = typer.Option(10, "--limit", min=1, help="How many recent benchmark runs to show"),
) -> None:
    settings = Settings()
    entries = _load_benchmark_history(settings=settings, limit=limit)
    payload = {"count": len(entries), "entries": entries}
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(f"Benchmark History (latest {len(entries)})")
    table = _render_table(
        rows=entries,
        columns=["timestamp", "successes", "failures", "quality_gate_failures", "avg_latency_s", "avg_reference_count", "file"],
        headers=["Timestamp", "OK", "Fail", "QGate", "Avg s", "Avg refs", "File"],
    )
    typer.echo(table)


@app.command("benchmark-compare")
def benchmark_compare(
    file_a: str = typer.Option("", "--file-a", help="First benchmark JSON file path"),
    file_b: str = typer.Option("", "--file-b", help="Second benchmark JSON file path"),
    as_json: bool = typer.Option(False, "--json", help="Print benchmark comparison as JSON"),
) -> None:
    settings = Settings()
    path_a, path_b = _resolve_compare_paths(settings=settings, file_a=file_a, file_b=file_b)
    data_a = json.loads(path_a.read_text(encoding="utf-8-sig"))
    data_b = json.loads(path_b.read_text(encoding="utf-8-sig"))
    summary_a = data_a.get("summary", {})
    summary_b = data_b.get("summary", {})
    delta = {
        "successes": int(summary_b.get("successes", 0)) - int(summary_a.get("successes", 0)),
        "failures": int(summary_b.get("failures", 0)) - int(summary_a.get("failures", 0)),
        "avg_latency_s": round(float(summary_b.get("avg_latency_s", 0.0)) - float(summary_a.get("avg_latency_s", 0.0)), 3),
        "avg_reference_count": round(
            float(summary_b.get("avg_reference_count", 0.0)) - float(summary_a.get("avg_reference_count", 0.0)),
            2,
        ),
        "total_tokens": int(summary_b.get("total_tokens", 0)) - int(summary_a.get("total_tokens", 0)),
        "quality_gate_failures": int(summary_b.get("quality_gate_failures", 0))
        - int(summary_a.get("quality_gate_failures", 0)),
    }
    payload = {
        "file_a": str(path_a),
        "file_b": str(path_b),
        "summary_a": summary_a,
        "summary_b": summary_b,
        "delta_b_minus_a": delta,
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo("Benchmark Compare (B - A)")
    typer.echo(f"A: {path_a}")
    typer.echo(f"B: {path_b}")
    typer.echo(
        _render_table(
            rows=[
                {"metric": "successes", "delta": delta["successes"]},
                {"metric": "failures", "delta": delta["failures"]},
                {"metric": "avg_latency_s", "delta": delta["avg_latency_s"]},
                {"metric": "avg_reference_count", "delta": delta["avg_reference_count"]},
                {"metric": "total_tokens", "delta": delta["total_tokens"]},
                {"metric": "quality_gate_failures", "delta": delta["quality_gate_failures"]},
            ],
            columns=["metric", "delta"],
            headers=["Metric", "Delta"],
        )
    )


@reports_app.command("list")
def reports_list(
    as_json: bool = typer.Option(False, "--json", help="Print reports list as JSON"),
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum report entries to return"),
    kind: str = typer.Option("all", "--kind", help="Filter by kind: all|file|session"),
) -> None:
    settings = Settings()
    assistant = ResearchAssistant(settings=settings)
    try:
        entries = _report_entries(assistant=assistant, settings=settings, limit=limit, kind=kind)
        payload = {"count": len(entries), "entries": entries}
        if as_json:
            typer.echo(json.dumps(payload, indent=2))
            return
        typer.echo(f"Reports ({len(entries)})")
        typer.echo(
            _render_table(
                rows=entries,
                columns=["ref", "kind", "updated_at", "report_chars", "query"],
                headers=["Ref", "Kind", "Updated", "Chars", "Query"],
            )
        )
    finally:
        assistant.close()


@reports_app.command("show")
def reports_show(
    ref: str = typer.Argument("latest", help="Report reference, e.g. latest, session:<id>, file:<name>"),
    as_json: bool = typer.Option(False, "--json", help="Print report metadata + markdown as JSON"),
    preview_lines: int = typer.Option(40, "--preview-lines", min=5, help="Number of lines to show in terminal preview"),
    full: bool = typer.Option(False, "--full", help="Print full markdown instead of preview"),
    kind: str = typer.Option("all", "--kind", help="For latest only: all|file|session"),
) -> None:
    settings = Settings()
    assistant = ResearchAssistant(settings=settings)
    try:
        resolved = _resolve_report_ref(assistant=assistant, settings=settings, ref=ref, kind=kind)
        if as_json:
            typer.echo(json.dumps(resolved, indent=2))
            return
        markdown = resolved["markdown"]
        line_count = len(markdown.splitlines())
        typer.echo(
            f"ref={resolved.get('ref','')} kind={resolved.get('kind','')} "
            f"query={resolved.get('query','')} lines={line_count}"
        )
        typer.echo("")
        if full or line_count <= preview_lines:
            typer.echo(markdown)
            return
        preview = "\n".join(markdown.splitlines()[:preview_lines])
        typer.echo(preview)
        typer.echo("")
        typer.echo(f"... truncated preview ({preview_lines}/{line_count} lines). Use --full to print all.")
    finally:
        assistant.close()


@reports_app.command("open")
def reports_open(
    ref: str = typer.Argument("latest", help="Report reference, e.g. latest, session:<id>, file:<name>"),
    kind: str = typer.Option("all", "--kind", help="For latest only: all|file|session"),
) -> None:
    settings = Settings()
    assistant = ResearchAssistant(settings=settings)
    try:
        resolved = _resolve_report_ref(assistant=assistant, settings=settings, ref=ref, kind=kind)
        path = _materialize_report_path(resolved=resolved, settings=settings)
        webbrowser.open(path.resolve().as_uri())
        typer.echo(f"Opened {path}")
    finally:
        assistant.close()


@reports_app.command("export")
def reports_export(
    from_ref: str = typer.Option("latest", "--from", help="Report reference, e.g. latest, session:<id>, file:<name>"),
    to: str = typer.Option("md", "--to", help="Export format: md|html|txt|pdf"),
    output: str = typer.Option("", "--output", "-o", help="Output file path"),
    kind: str = typer.Option("all", "--kind", help="For latest only: all|file|session"),
) -> None:
    settings = Settings()
    assistant = ResearchAssistant(settings=settings)
    try:
        resolved = _resolve_report_ref(assistant=assistant, settings=settings, ref=from_ref, kind=kind)
        fmt = to.strip().lower()
        if fmt not in {"md", "html", "txt", "pdf"}:
            raise ValueError("Unsupported export format. Use one of: md, html, txt, pdf.")
        if output:
            out_path = Path(output)
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = Path(settings.reports_directory) / f"export_{stamp}.{fmt}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        markdown = resolved["markdown"]
        if fmt == "md":
            out_path.write_text(markdown, encoding="utf-8")
        elif fmt == "txt":
            out_path.write_text(markdown, encoding="utf-8")
        elif fmt == "pdf":
            _markdown_to_pdf(path=out_path, markdown=markdown, title=resolved.get("query", "Research Report"))
        else:
            out_path.write_text(_markdown_to_html(markdown, title=resolved.get("query", "Research Report")), encoding="utf-8")
        typer.echo(f"Exported report to {out_path}")
    finally:
        assistant.close()


if __name__ == "__main__":
    app()


def main() -> None:
    app()


def _count_references(report: str) -> int:
    lines = report.splitlines()
    return sum(1 for line in lines if line.strip().startswith("[") and "] " in line)


def _load_benchmark_queries(path: str) -> list[str]:
    if path:
        raw = Path(path).read_text(encoding="utf-8")
        rows = [line.strip() for line in raw.splitlines() if line.strip()]
        if rows:
            return rows
    return [
        "Impact of AI on education",
        "Benefits and risks of renewable energy transition",
        "How quantum computing may affect cybersecurity",
        "Compare retrieval-augmented generation frameworks",
        "Trends in electric vehicle battery technology",
    ]


def _classify_error_type(error: str) -> str:
    lowered = (error or "").lower()
    if "429" in lowered or "too many requests" in lowered:
        return "rate_limit"
    if "timeout" in lowered:
        return "timeout"
    if "connectionerror" in lowered or "failed to establish a new connection" in lowered:
        return "connection_error"
    if "keyerror" in lowered:
        return "schema_error"
    if "valueerror" in lowered:
        return "value_error"
    if ":" in error:
        return error.split(":", 1)[0].strip().lower()
    return "unknown"


def _total_tokens_from_metrics(metrics: dict) -> int:
    return sum(int((row or {}).get("total_tokens", 0)) for row in (metrics or {}).values())


def _extract_quality(result: dict) -> dict:
    return ((result.get("metadata", {}) or {}).get("quality", {}) or {})


def _should_try_adaptive(settings: Settings, quality_passed: bool) -> bool:
    if quality_passed:
        return False
    return bool(getattr(settings, "adaptive_depth_enabled", True))


def _within_budget_headroom(duration_s: float, query_total_tokens: int, max_seconds: float, max_tokens: int) -> bool:
    if max_seconds > 0 and duration_s >= max_seconds:
        return False
    if max_tokens > 0 and query_total_tokens >= max_tokens:
        return False
    return True


def _run_adaptive_passes(
    assistant: ResearchAssistant,
    query: str,
    base_result: dict,
    base_duration_s: float,
    before_tokens: int,
    settings: Settings,
) -> tuple[dict, float, int]:
    current_result = base_result
    total_duration_s = base_duration_s
    max_passes = max(1, int(getattr(settings, "adaptive_max_passes", 1)))
    sq_inc = max(0, int(getattr(settings, "adaptive_sub_questions_increment", 1)))
    it_inc = max(0, int(getattr(settings, "adaptive_iterations_increment", 1)))
    orig_sub_q = int(getattr(assistant.settings, "max_sub_questions", 1))
    orig_iters = int(getattr(assistant.settings, "max_research_iterations", 1))
    max_tokens = int(getattr(settings, "max_total_tokens_per_query", 0))
    max_seconds = float(getattr(settings, "max_seconds_per_query", 0.0))

    try:
        for n in range(1, max_passes + 1):
            assistant.settings.max_sub_questions = orig_sub_q + (sq_inc * n)
            assistant.settings.max_research_iterations = orig_iters + (it_inc * n)
            started = time.perf_counter()
            next_result = assistant.research(query)
            total_duration_s += time.perf_counter() - started
            current_result = next_result
            quality = _extract_quality(current_result)
            if bool(quality.get("passed", True)):
                break
            after_tokens = _total_tokens_from_metrics(assistant.get_llm_metrics())
            query_total_tokens = max(0, after_tokens - before_tokens)
            if not _within_budget_headroom(total_duration_s, query_total_tokens, max_seconds, max_tokens):
                break
    finally:
        assistant.settings.max_sub_questions = orig_sub_q
        assistant.settings.max_research_iterations = orig_iters

    after_tokens = _total_tokens_from_metrics(assistant.get_llm_metrics())
    query_total_tokens = max(0, after_tokens - before_tokens)
    return current_result, total_duration_s, query_total_tokens


def _save_benchmark_payload(payload: dict, settings: Settings) -> Path:
    reports_dir = Path(settings.reports_directory)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    history_path = reports_dir / f"benchmark_{stamp}.json"
    latest_path = reports_dir / "benchmark_latest.json"
    content = json.dumps(payload, indent=2)
    history_path.write_text(content, encoding="utf-8")
    latest_path.write_text(content, encoding="utf-8")
    return history_path


def _load_benchmark_history(settings: Settings, limit: int) -> list[dict]:
    reports_dir = Path(settings.reports_directory)
    if not reports_dir.exists():
        return []
    files = sorted(
        [p for p in reports_dir.glob("benchmark_*.json") if p.name != "benchmark_latest.json"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    rows = []
    for path in files[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        summary = payload.get("summary", {})
        rows.append(
            {
                "file": str(path),
                "timestamp": path.stem.replace("benchmark_", ""),
                "queries_run": int(summary.get("queries_run", 0)),
                "successes": int(summary.get("successes", 0)),
                "failures": int(summary.get("failures", 0)),
                "quality_gate_failures": int(summary.get("quality_gate_failures", 0)),
                "avg_latency_s": float(summary.get("avg_latency_s", 0.0)),
                "avg_reference_count": float(summary.get("avg_reference_count", 0.0)),
                "total_tokens": int(summary.get("total_tokens", 0)),
            }
        )
    return rows


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
        raise ValueError("Need at least two benchmark history files for comparison.")
    return selected[1], selected[0]


def _report_entries(assistant: ResearchAssistant, settings: Settings, limit: int, kind: str = "all") -> list[dict]:
    k = (kind or "all").strip().lower()
    if k not in {"all", "file", "session"}:
        raise ValueError("Invalid kind. Use one of: all, file, session.")
    session_entries = assistant.list_completed_reports(limit=limit)
    reports_dir = Path(settings.reports_directory)
    file_entries: list[dict] = []
    if reports_dir.exists():
        md_files = sorted(reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        for path in md_files:
            updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            file_entries.append(
                {
                    "ref": f"file:{path.name}",
                    "kind": "file",
                    "session_id": "",
                    "query": path.stem,
                    "created_at": updated_at,
                    "updated_at": updated_at,
                    "report_chars": int(path.stat().st_size),
                    "has_report": True,
                    "path": str(path),
                }
            )
    if k == "file":
        all_entries = file_entries
    elif k == "session":
        all_entries = session_entries
    else:
        all_entries = file_entries + session_entries
    if k == "all":
        all_entries.sort(
            key=lambda row: -(datetime.fromisoformat(row.get("updated_at", "1970-01-01T00:00:00+00:00")).timestamp() if row.get("updated_at") else 0.0),
        )
    else:
        all_entries.sort(key=lambda row: row.get("updated_at", ""), reverse=True)
    return all_entries[:limit]


def _resolve_report_ref(assistant: ResearchAssistant, settings: Settings, ref: str, kind: str = "all") -> dict:
    target = (ref or "latest").strip()
    if target in {"latest", ""}:
        entries = _report_entries(assistant=assistant, settings=settings, limit=1, kind=kind)
        if not entries:
            raise ValueError("No reports found.")
        target = entries[0]["ref"]

    if target.startswith("session:"):
        sid = target.split(":", 1)[1].strip()
        row = assistant.get_result(sid)
        status = row.get("status", "")
        result = row.get("result", {}) or {}
        markdown = str(result.get("final_report", "") or "")
        if status != "completed" or not markdown:
            raise ValueError(f"Session report not available: {sid}")
        query = result.get("query", "") or ""
        if not query:
            query = row.get("session_id", sid)
        return {
            "ref": f"session:{sid}",
            "kind": "session",
            "query": query,
            "updated_at": "",
            "markdown": markdown,
        }

    if target.startswith("file:"):
        name = target.split(":", 1)[1].strip()
        path = Path(settings.reports_directory) / name
        if not path.exists():
            raise ValueError(f"Report file not found: {path}")
        return {
            "ref": f"file:{name}",
            "kind": "file",
            "query": path.stem,
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            "markdown": path.read_text(encoding="utf-8"),
            "path": str(path),
        }
    raise ValueError("Unsupported ref. Use latest, session:<id>, or file:<name>.")


def _materialize_report_path(resolved: dict, settings: Settings) -> Path:
    if resolved.get("kind") == "file" and resolved.get("path"):
        return Path(resolved["path"])
    reports_dir = Path(settings.reports_directory)
    reports_dir.mkdir(parents=True, exist_ok=True)
    safe_ref = resolved.get("ref", "session").replace(":", "_")
    out = reports_dir / f"{safe_ref}.md"
    out.write_text(resolved.get("markdown", ""), encoding="utf-8")
    return out


def _markdown_to_html(markdown: str, title: str) -> str:
    body = _render_markdown_html_body(markdown or "")
    safe_title = html.escape(title or "Research Report")
    return (
        "<!doctype html>\n"
        "<html><head><meta charset='utf-8'>"
        f"<title>{safe_title}</title>"
        "<style>"
        "body{font-family:'Segoe UI',Arial,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;line-height:1.6;color:#222}"
        "h1{font-size:2rem;margin:0 0 1rem 0}"
        "h2,h3,h4,h5,h6{margin:1.25rem 0 0.6rem 0}"
        "p{margin:0.55rem 0}"
        "ul,ol{margin:0.4rem 0 0.9rem 1.2rem}"
        "li{margin:0.25rem 0}"
        "code{background:#f3f4f6;border-radius:4px;padding:0.1rem 0.3rem;font-family:Consolas,'Courier New',monospace}"
        "pre{white-space:pre-wrap;background:#f6f8fa;padding:1rem;border-radius:8px;overflow-x:auto}"
        "blockquote{border-left:4px solid #d0d7de;padding:0.15rem 0.8rem;margin:0.8rem 0;color:#444;background:#fafbfc}"
        "a{color:#0a58ca;text-decoration:underline;cursor:pointer;word-break:break-all}"
        "a:hover{opacity:0.9}"
        ".citation{font-size:0.95rem}"
        "</style>"
        "</head><body>"
        f"<h1>{safe_title}</h1>"
        f"{body}"
        "</body></html>"
    )


def _render_markdown_html_body(markdown: str) -> str:
    import re

    lines = (markdown or "").splitlines()
    chunks: list[str] = []
    paragraph_parts: list[str] = []
    list_kind = ""
    list_items: list[str] = []
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_parts
        if paragraph_parts:
            chunks.append(f"<p>{_format_inline(' '.join(paragraph_parts).strip())}</p>")
            paragraph_parts = []

    def flush_list() -> None:
        nonlocal list_kind, list_items
        if list_items:
            items = "".join(f"<li>{item}</li>" for item in list_items)
            chunks.append(f"<{list_kind}>{items}</{list_kind}>")
        list_kind = ""
        list_items = []

    def flush_code() -> None:
        nonlocal code_lines
        if code_lines:
            chunks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
            code_lines = []

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            flush_list()
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
                code_lines = []
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            flush_list()
            level = len(heading.group(1))
            chunks.append(f"<h{level}>{_format_inline(heading.group(2).strip())}</h{level}>")
            continue

        unordered = re.match(r"^\s*[-*+]\s+(.*)$", line)
        ordered = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if unordered or ordered:
            flush_paragraph()
            kind = "ol" if ordered else "ul"
            content = (ordered.group(1) if ordered else unordered.group(1)).strip()
            if list_kind and list_kind != kind:
                flush_list()
            list_kind = kind
            list_items.append(_format_inline(content))
            continue

        flush_list()
        if stripped.startswith(">"):
            flush_paragraph()
            chunks.append(f"<blockquote>{_format_inline(stripped[1:].strip())}</blockquote>")
            continue
        if re.match(r"^\[\d+\]\s+", stripped):
            flush_paragraph()
            chunks.append(f"<p class='citation'>{_format_inline(stripped)}</p>")
            continue

        paragraph_parts.append(stripped)

    flush_paragraph()
    flush_list()
    if in_code:
        flush_code()

    return "".join(chunks) if chunks else "<p></p>"


def _format_inline(text: str) -> str:
    import re

    if not text:
        return ""
    link_pattern = re.compile(r"\[([^\]]+)\]\(([A-Za-z][A-Za-z0-9+.-]*://[^)\s]+)\)")
    parts: list[str] = []
    last = 0
    for match in link_pattern.finditer(text):
        parts.append(_auto_link_urls(html.escape(text[last : match.start()])))
        label = html.escape(match.group(1))
        url = html.escape(match.group(2), quote=True)
        parts.append(f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>')
        last = match.end()
    parts.append(_auto_link_urls(html.escape(text[last:])))
    return "".join(parts)


def _auto_link_urls(escaped_text: str) -> str:
    import re

    url_pattern = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://[^\s<]+)")

    def repl(match):
        token = match.group(1)
        url = token
        trailer = ""
        while url and url[-1] in ".,);]":
            trailer = url[-1] + trailer
            url = url[:-1]
        return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>{trailer}'

    return url_pattern.sub(repl, escaped_text)


def _markdown_to_pdf(path: Path, markdown: str, title: str) -> None:
    pages = _paginate_pdf_lines(_markdown_lines_for_pdf(markdown=markdown, title=title), max_lines_per_page=52)
    pdf_bytes = _build_basic_pdf(pages)
    path.write_bytes(pdf_bytes)


def _markdown_lines_for_pdf(markdown: str, title: str) -> list[str]:
    lines = [title or "Research Report", ""]
    for line in (markdown or "").splitlines():
        text = line.replace("\t", "    ").strip()
        if not text:
            lines.append("")
            continue
        lines.extend(_wrap_pdf_line(text, width=95))
    return lines


def _wrap_pdf_line(text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    rows: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            rows.append(current)
            current = word
    rows.append(current)
    return rows


def _paginate_pdf_lines(lines: list[str], max_lines_per_page: int) -> list[list[str]]:
    if not lines:
        return [[""]]
    pages: list[list[str]] = []
    for idx in range(0, len(lines), max_lines_per_page):
        pages.append(lines[idx : idx + max_lines_per_page])
    return pages


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_basic_pdf(pages: list[list[str]]) -> bytes:
    objects: dict[int, bytes] = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    page_ids: list[int] = []
    next_obj = 3
    for _ in pages:
        page_ids.append(next_obj)
        next_obj += 2
    font_id = next_obj

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[2] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode("utf-8")

    for index, page_lines in enumerate(pages):
        page_id = page_ids[index]
        content_id = page_id + 1
        stream_lines = ["BT", "/F1 11 Tf", "50 770 Td", "14 TL"]
        for line in page_lines:
            stream_lines.append(f"({_pdf_escape(line)}) Tj")
            stream_lines.append("T*")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("utf-8")
        objects[content_id] = b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("utf-8")

    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    max_obj = max(objects.keys())

    doc = bytearray()
    doc.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0] * (max_obj + 1)
    for obj_id in range(1, max_obj + 1):
        offsets[obj_id] = len(doc)
        body = objects[obj_id]
        doc.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        doc.extend(body)
        doc.extend(b"\nendobj\n")

    xref_offset = len(doc)
    doc.extend(f"xref\n0 {max_obj + 1}\n".encode("ascii"))
    doc.extend(b"0000000000 65535 f \n")
    for obj_id in range(1, max_obj + 1):
        doc.extend(f"{offsets[obj_id]:010d} 00000 n \n".encode("ascii"))
    doc.extend(f"trailer\n<< /Size {max_obj + 1} /Root 1 0 R >>\n".encode("ascii"))
    doc.extend(f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    return bytes(doc)


def _render_table(rows: list[dict], columns: list[str], headers: list[str]) -> str:
    if not rows:
        return "(no data)"
    widths = []
    for idx, col in enumerate(columns):
        max_cell = max(len(str((row or {}).get(col, ""))) for row in rows)
        widths.append(max(len(headers[idx]), max_cell))

    def fmt(values: list[str]) -> str:
        parts = []
        for i, value in enumerate(values):
            parts.append(str(value).ljust(widths[i]))
        return " | ".join(parts)

    top = fmt(headers)
    sep = "-+-".join("-" * w for w in widths)
    body = [fmt([str((row or {}).get(col, "")) for col in columns]) for row in rows]
    return "\n".join([top, sep] + body)


def _is_synthetic_benchmark_file(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    return int(summary.get("total_llm_calls", 0) or 0) == 0
