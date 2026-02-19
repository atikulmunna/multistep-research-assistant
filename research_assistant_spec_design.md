# Research Assistant with Multi-Step Reasoning
## Unified Specification and Design Document

**Version:** 1.1  
**Date:** February 18, 2026  
**Status:** Ready for MVP Implementation (Phase 1)

---

## 1. Executive Summary

This project builds an AI-powered research assistant that autonomously performs multi-step research for complex queries. The system uses LangChain for model/tool integration and LangGraph for stateful workflow orchestration across planning, gathering, analysis, synthesis, and report generation.

---

## 2. Objectives and Success Criteria

### 2.1 Primary Objectives
- Automate multi-step research for complex questions
- Produce structured, citation-backed reports
- Provide a reusable agentic workflow framework
- Support iterative refinement and gap closure

### 2.2 Success Criteria
- Decompose complex queries into 3-5 sub-questions
- Achieve relevance score >= 0.80 on retained evidence
- Generate coherent reports synthesizing multiple sources
- Complete standard research tasks in <= 5 minutes
- Support up to 5 refinement iterations per session

---

## 3. Scope

### 3.1 Phase 1 (MVP, in scope)
- Query decomposition and planning
- Web search integration
- Multi-format parsing (HTML, PDF, MD, TXT)
- Relevance scoring and gap detection
- Contradiction and source credibility assessment (basic)
- Markdown report generation with citations
- CLI interface with progress updates

### 3.2 Phase 2 (future)
- Web API and web UI
- Session resume, follow-up Q&A
- Enhanced contradiction/credibility models
- Export to PDF/HTML

### 3.3 Out of Scope (current)
- Real-time multi-user collaboration
- Paywalled/proprietary DB integrations
- Non-English query support
- Mobile app

---

## 4. Functional Requirements

- **FR-1 Query Understanding:** Accept natural-language research questions; classify query type and scope.
- **FR-2 Decomposition:** Produce 3-5 answerable, non-overlapping sub-questions.
- **FR-3 Information Gathering:** Search multiple sources and retain metadata, content, and citations.
- **FR-4 Analysis:** Score relevance, detect gaps, identify contradictions, and assess source credibility.
- **FR-5 Sufficiency Decision:** Determine whether evidence is sufficient; generate additional sub-questions for gaps.
- **FR-6 Synthesis & Reporting:** Produce structured report with executive summary, findings, gaps, and references.
- **FR-7 User Interaction:** Show progress updates; allow async status polling in API mode (Phase 2).

---

## 5. Non-Functional Requirements

### 5.1 Performance
- Query decomposition <= 10s
- Each search iteration <= 15s
- Report generation <= 30s
- Total standard query runtime <= 5 minutes

### 5.2 Reliability
- Graceful handling of API failures
- Retry logic for transient failures
- Progress persistence to avoid data loss

### 5.3 Security and Privacy
- API keys via environment variables only
- Sensitive info redacted from logs
- No raw user data logging without consent

### 5.4 Maintainability
- Modular architecture
- Type-checked models and documented interfaces
- Comprehensive structured logging

---

## 6. Measurable Quality Metrics

- **Relevance Score (0-1):** mean similarity between sub-question and extracted claims, calibrated by LLM rubric + embedding similarity.  
  Pass threshold: `>= 0.80` on retained claims.
- **Credibility Score (0-1):** weighted score from domain trust, recency, author/source transparency, and cross-source agreement.  
  Reported per source and aggregated per section.
- **Contradiction Precision:** fraction of flagged contradictions that are true conflicts in evaluation set.  
  MVP target: `>= 0.70`.
- **Coverage Score (0-1):** fraction of sub-questions with at least one high-relevance claim (`>= 0.80`) from >=2 sources.

---

## 7. Architecture

### 7.1 High-Level Components
- Interface Layer: CLI (Phase 1), Web/API (Phase 2)
- Orchestration Layer: LangGraph workflow
- Services: Search, LLM, Parser, Citation, Report Formatting
- External Dependencies: OpenAI/Anthropic, search provider(s), optional vector store

### 7.2 Request Flow
1. User submits query
2. Workflow initializes state
3. Nodes update shared state with append-safe reducers
4. Sufficiency gate decides `gather` loop or `synthesize`
5. Report generated and returned/exported

---

## 8. Data Models

### 8.1 Core Types

```python
from typing import TypedDict, List, Dict, Optional
from datetime import datetime

class SubQuestion(TypedDict):
    question: str
    priority: int
    answered: bool
    parent_question: Optional[str]

class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str
    source: str
    timestamp: datetime
    relevance_score: float

class AnalyzedSource(TypedDict):
    sub_question: str
    key_information: List[str]
    citations: List[str]
    relevance_score: float
    credibility_score: float
    contradictions: List[str]
```

### 8.2 LangGraph State (normalized)

```python
class ResearchState(TypedDict):
    query: str
    analyzed_query: Dict
    sub_questions: List[SubQuestion]
    current_question_idx: int

    search_results: Dict[str, List[SearchResult]]
    raw_documents: Dict[str, List[Dict]]

    analyzed_content: List[AnalyzedSource]
    identified_gaps: List[str]
    contradictions: List[Dict]

    report_sections: Dict[str, str]
    final_report: str

    metadata: Dict
    errors: List[str]
    total_tokens_used: int
    execution_time: float
    iteration_count: int
```

---

## 9. Workflow Design (LangGraph)

### 9.1 Nodes
- `analyze_query`
- `plan_research`
- `gather_information`
- `analyze_content`
- `synthesize_information`
- `generate_report`

### 9.2 Loop and Exit Conditions
- `max_research_iterations = 5`
- Loop continues while:
  - uncovered sub-questions remain, or
  - coverage score < 0.80, or
  - critical gaps exist
- Hard stop at iteration cap with warning in report metadata

### 9.3 Corrected Node Contracts

```python
def gather_information(state: ResearchState) -> ResearchState:
    sq = state["sub_questions"][state["current_question_idx"]]["question"]
    results = search_service.search(sq, max_results=5)

    parsed = []
    for r in results:
        content = search_service.fetch_url(r["url"])
        doc_type = document_parser.detect_type(r["url"], content)
        parsed.append(document_parser.parse(content, doc_type))

    state["search_results"][sq] = results
    state["raw_documents"][sq] = parsed
    state["current_question_idx"] += 1
    return state
```

```python
def should_search(state: ResearchState) -> str:
    if state["current_question_idx"] < len(state["sub_questions"]):
        return "gather"
    return "analyze"
```

---

## 10. Services and Interfaces

### 10.1 Search Service
- `search(query, max_results=5) -> List[SearchResult]`
- `fetch_url(url) -> str`
- `batch_search(queries) -> Dict[str, List[SearchResult]]`

### 10.2 Document Parser
- `detect_type(url, content) -> str`
- `parse(content, doc_type) -> Dict`
- Supports `html`, `pdf`, `md`, `txt`

### 10.3 LLM Service
- `invoke(prompt, context=None) -> str`
- `invoke_structured(prompt, schema) -> Dict`
- token tracking, retries, and response validation

### 10.4 Citation Manager
- canonical source IDs
- inline markers and bibliography generation

---

## 11. API and CLI

### 11.1 Public API
- `research(query: str) -> ResearchReport`
- `research_async(query: str) -> str`
- `get_progress(session_id: str) -> Dict`
- `export_report(report, path, format="md")`

### 11.2 CLI

```bash
research-assistant "impact of AI on education"
research-assistant "climate change solutions" --depth comprehensive --format md
research-assistant --interactive
```

---

## 12. Configuration

```python
class Settings(BaseSettings):
    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4000

    search_provider: str = "tavily"
    max_search_results: int = 5
    max_sources_per_question: int = 5

    max_sub_questions: int = 5
    max_research_iterations: int = 5
    min_relevance_score: float = 0.8

    default_output_format: str = "markdown"
    reports_directory: str = "./reports"
```

---

## 13. Testing Strategy and Traceability

### 13.1 Test Coverage Targets
- Unit coverage > 80%
- All workflow branches covered in integration tests
- Performance benchmarks for query decomposition and full run

### 13.2 FR to Test Mapping
- FR-1/FR-2 -> query analysis and decomposition unit tests
- FR-3 -> search/parser integration tests
- FR-4/FR-5 -> analysis, contradiction, and sufficiency decision tests
- FR-6 -> report formatting + citation completeness tests
- NFR performance -> benchmark tests (`<10s`, `<15s`, `<30s`, `<300s`)

---

## 14. Risks and Mitigation

- API rate limits: caching, retries, adaptive backoff
- Cost overruns: per-session token budget and early-stop thresholds
- Hallucinations: citation-required generation and unsupported-claim filtering
- Poor source quality: credibility thresholding and source diversity requirements

---

## 15. Acceptance Criteria

1. All Phase 1 FRs implemented and validated by tests
2. Quality gates met (relevance >= 0.80, citation completeness 100%)
3. Runtime NFRs met for standard queries
4. 10 diverse end-to-end query demos completed successfully
5. Documentation complete: README, config, usage, architecture notes

---

## 16. Document Control

- **Authors:** AI Research + Architecture Team  
- **Last Updated:** February 18, 2026  
- **Next Review:** March 18, 2026

