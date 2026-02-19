from typing import Dict, List
from urllib.parse import urlparse

import requests

from .state import ResearchState
from ..utils.contradictions import detect_contradictions


def analyze_query_node(state: ResearchState) -> Dict:
    _emit_progress(state, stage="analyze_query")
    llm = state["metadata"]["services"]["llm"]
    analyzed = llm.analyze_query(state["query"])
    return {"analyzed_query": analyzed}


def plan_research_node(state: ResearchState) -> Dict:
    _emit_progress(state, stage="plan_research")
    llm = state["metadata"]["services"]["llm"]
    settings = state["metadata"]["settings"]
    sub_questions = llm.decompose_query(state["query"], max_questions=settings.max_sub_questions)
    _emit_progress(
        state,
        stage="plan_research_done",
        sub_questions_total=len(sub_questions),
        sub_questions_done=0,
    )
    return {"sub_questions": sub_questions, "current_question_idx": 0}


def gather_information_node(state: ResearchState) -> Dict:
    settings = state["metadata"]["settings"]
    services = state["metadata"]["services"]
    current_idx = state["current_question_idx"]
    total = len(state["sub_questions"])

    _emit_progress(
        state,
        stage="gather_information",
        sub_questions_total=total,
        sub_questions_done=min(current_idx, total),
        current_sub_question=state["sub_questions"][current_idx]["question"] if current_idx < total else "",
    )

    if current_idx >= len(state["sub_questions"]):
        return {}

    sub_question = state["sub_questions"][current_idx]["question"]
    results = services["search"].search(sub_question, max_results=settings.max_search_results)
    results = _dedupe_results_by_url(results)

    parsed_docs: List[Dict] = []
    errors = list(state.get("errors", []))
    warnings = list(state["metadata"].get("warnings", []))
    for row in results:
        url = row.get("url", "")
        if not url:
            continue
        try:
            content = services["search"].fetch_url(url)
            doc_type = services["parser"].detect_type(url, content)
            parsed_docs.append(services["parser"].parse(content, doc_type, url))
        except (requests.RequestException, TimeoutError) as exc:
            errors.append(f"fetch_failed:{url}:{type(exc).__name__}")
            warnings.append(f"Skipping inaccessible URL: {url}")
            continue
        except Exception as exc:
            errors.append(f"parse_failed:{url}:{type(exc).__name__}")
            warnings.append(f"Skipping unparseable URL: {url}")
            continue

    next_search = dict(state["search_results"])
    next_search[sub_question] = results

    next_docs = dict(state["raw_documents"])
    next_docs[sub_question] = parsed_docs
    metadata = dict(state["metadata"])
    metadata["warnings"] = warnings

    return {
        "search_results": next_search,
        "raw_documents": next_docs,
        "current_question_idx": current_idx + 1,
        "errors": errors,
        "metadata": metadata,
    }


def analyze_content_node(state: ResearchState) -> Dict:
    _emit_progress(
        state,
        stage="analyze_content",
        analyzed_items=len(state["analyzed_content"]),
        iteration=state["iteration_count"] + 1,
    )
    llm = state["metadata"]["services"]["llm"]
    existing = list(state["analyzed_content"])
    analyzed_questions = {item["sub_question"] for item in existing}
    contradictions = list(state["contradictions"])

    for sq in state["sub_questions"]:
        question = sq["question"]
        if question in analyzed_questions:
            continue
        docs = state["raw_documents"].get(question, [])
        extracted = llm.extract_key_info(question, docs)
        key_information = _as_string_list(extracted.get("key_information", []))
        citations = _normalize_citations(
            candidate=_as_url_list(extracted.get("citations", [])),
            docs=docs,
        )
        contradiction_items = _as_string_list(extracted.get("contradictions", []))
        relevance = _estimate_relevance(key_information, question)
        heuristic_contradictions = detect_contradictions(key_information)
        combined_contradictions = contradiction_items + heuristic_contradictions

        item = {
            "sub_question": question,
            "key_information": key_information,
            "citations": citations,
            "relevance_score": relevance,
            "credibility_score": _estimate_credibility(citations),
            "contradictions": combined_contradictions,
        }
        existing.append(item)
        for c in item["contradictions"]:
            contradictions.append({"sub_question": question, "detail": c})

    gaps = [
        row["sub_question"]
        for row in existing
        if row["relevance_score"] < state["metadata"]["settings"].min_relevance_score
    ]
    return {
        "analyzed_content": existing,
        "identified_gaps": gaps,
        "contradictions": contradictions,
        "iteration_count": state["iteration_count"] + 1,
    }


def synthesize_information_node(state: ResearchState) -> Dict:
    _emit_progress(state, stage="synthesize")
    llm = state["metadata"]["services"]["llm"]
    sections = {
        "Findings Overview": llm.synthesize_section("Findings Overview", state["analyzed_content"]),
        "Evidence by Sub-question": llm.synthesize_section("Evidence by Sub-question", state["analyzed_content"]),
    }
    return {"report_sections": sections}


def generate_report_node(state: ResearchState) -> Dict:
    _emit_progress(state, stage="generate_report")
    services = state["metadata"]["services"]
    citation_manager = services["citation"]

    key_findings: List[str] = []
    for item in state["analyzed_content"]:
        for citation in item["citations"]:
            citation_manager.add_source(citation)
        key_findings.extend(item["key_information"][:1])

    settings = state["metadata"]["settings"]
    gaps = list(state["identified_gaps"])
    references = citation_manager.bibliography()
    unique_domains = _unique_domains_from_urls(references)
    min_domains = int(getattr(settings, "min_unique_source_domains", 2))
    min_refs = int(getattr(settings, "min_reference_count", 3))
    if unique_domains and len(unique_domains) < min_domains:
        gaps.append(
            f"Source diversity warning: only {len(unique_domains)} unique source domain(s), "
            f"below configured minimum of {min_domains}."
        )
    if len(references) < min_refs:
        gaps.append(
            f"Evidence coverage warning: only {len(references)} reference(s), "
            f"below configured minimum of {min_refs}."
        )

    quality_checks: List[str] = []
    if len(references) < min_refs:
        quality_checks.append("reference_count_below_min")
    if unique_domains and len(unique_domains) < min_domains:
        quality_checks.append("source_diversity_below_min")

    quality = {
        "passed": len(quality_checks) == 0,
        "checks_failed": quality_checks,
        "reference_count": len(references),
        "unique_source_domains": len(unique_domains),
        "min_reference_count": min_refs,
        "min_unique_source_domains": min_domains,
    }

    metadata = dict(state["metadata"])
    metadata["quality"] = quality
    _emit_progress(
        state,
        stage="completed",
        sub_questions_done=len(state.get("sub_questions", [])),
        sub_questions_total=len(state.get("sub_questions", [])),
        iteration=state.get("iteration_count", 0),
        quality_passed=quality["passed"],
        references=quality["reference_count"],
        unique_domains=quality["unique_source_domains"],
    )

    summary = services["llm"].summarize_report(state["query"], state["report_sections"])
    markdown = services["formatter"].to_markdown(
        query=state["query"],
        summary=summary,
        sections=state["report_sections"],
        key_findings=key_findings[:8],
        gaps=gaps,
        references=references,
    )
    return {"final_report": markdown, "metadata": metadata}


def _estimate_relevance(claims: List[str], question: str) -> float:
    if not claims:
        return 0.0
    q_tokens = set(question.lower().split())
    c_tokens = set(" ".join(claims).lower().split())
    overlap = len(q_tokens & c_tokens)
    return max(0.4, min(0.98, overlap / max(1, len(q_tokens))))


def _estimate_credibility(citations: List[str]) -> float:
    if not citations:
        return 0.3
    known = sum(1 for url in citations if isinstance(url, str) and (url.startswith("https://") or url.startswith("mock://")))
    return max(0.5, min(0.95, known / len(citations)))


def _dedupe_results_by_url(results: List[Dict]) -> List[Dict]:
    seen = set()
    deduped = []
    for row in results:
        url = row.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(row)
    return deduped


def _unique_domains_from_urls(reference_lines: List[str]) -> set[str]:
    domains = set()
    for line in reference_lines:
        url = _extract_reference_url(line)
        if not url:
            continue
        if url.startswith("mock://"):
            domains.add("mock")
            continue
        try:
            host = urlparse(url).netloc.lower()
            if host:
                domains.add(host)
        except Exception:
            continue
    return domains


def _as_string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for row in value:
        if isinstance(row, str) and row.strip():
            out.append(row.strip())
    return out


def _as_url_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for row in value:
        if isinstance(row, str) and row.strip():
            normalized = _normalize_url(row.strip())
            if normalized:
                out.append(normalized)
            continue
        if isinstance(row, dict):
            for key in ("url", "link", "source"):
                item = row.get(key)
                if isinstance(item, str) and item.strip():
                    normalized = _normalize_url(item.strip())
                    if normalized:
                        out.append(normalized)
                    break
    return out


def _normalize_citations(candidate: List[str], docs: List[Dict]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for item in candidate:
        normalized = _normalize_url(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    if deduped:
        return deduped

    # Fallback to parser/source URLs when LLM returns placeholder citations.
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        normalized = _normalize_url(str(doc.get("source_url", "")).strip())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _normalize_url(value: str) -> str:
    token = (value or "").strip()
    if not token:
        return ""
    # Accept any URL scheme (http/https/mock/etc.). Bare placeholders are rejected.
    if "://" in token and " " not in token:
        return token
    if token.startswith("www.") and " " not in token:
        return f"https://{token}"
    return ""


def _extract_reference_url(line: str) -> str:
    parts = (line or "").split(" ", 1)
    if len(parts) != 2:
        return ""
    token = parts[1].strip()
    if not token:
        return ""
    if token.startswith("[") and "](" in token and token.endswith(")"):
        open_paren = token.rfind("(")
        close_paren = token.rfind(")")
        if open_paren != -1 and close_paren > open_paren:
            return token[open_paren + 1 : close_paren].strip()
    return token


def _emit_progress(state: ResearchState, stage: str, **fields: object) -> None:
    metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
    callback = metadata.get("progress_callback")
    if not callable(callback):
        return
    payload = {"stage": stage}
    payload.update(fields)
    try:
        callback(payload)
    except Exception:
        return
