import ast
import json
import random
import re
import time
from typing import Any, Dict, List

import requests


class LLMService:
    def __init__(
        self,
        provider: str = "mock",
        model: str = "gpt-4.1-mini",
        api_key: str = "",
        groq_api_key: str = "",
        xai_api_key: str = "",
        openrouter_api_key: str = "",
        ollama_base_url: str = "http://127.0.0.1:11434",
        task_models: Dict[str, str] | None = None,
        fallback_enabled: bool = True,
        fallback_provider: str = "",
        fallback_model: str = "",
        second_fallback_provider: str = "",
        second_fallback_model: str = "",
        retry_max_attempts: int = 4,
        retry_base_delay_s: float = 1.0,
        retry_max_delay_s: float = 8.0,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.groq_api_key = groq_api_key
        self.xai_api_key = xai_api_key
        self.openrouter_api_key = openrouter_api_key
        self.ollama_base_url = (ollama_base_url or "http://127.0.0.1:11434").rstrip("/")
        self.task_models = task_models or {}
        self.fallback_enabled = fallback_enabled
        self.fallback_provider = fallback_provider
        self.fallback_model = fallback_model
        self.second_fallback_provider = second_fallback_provider
        self.second_fallback_model = second_fallback_model
        self.retry_max_attempts = max(1, int(retry_max_attempts))
        self.retry_base_delay_s = max(0.0, float(retry_base_delay_s))
        self.retry_max_delay_s = max(0.0, float(retry_max_delay_s))
        self.metrics: Dict[str, Dict[str, Any]] = {}

    def analyze_query(self, query: str) -> Dict[str, str]:
        if self.provider == "mock":
            q = query.lower()
            query_type = "comparative" if "compare" in q else "exploratory"
            scope = "broad" if len(query.split()) > 6 else "medium"
            return {"query_type": query_type, "scope": scope}
        prompt = (
            "Classify the research query. Return JSON with keys "
            "query_type (factual|comparative|exploratory) and scope (narrow|medium|broad).\n"
            f"Query: {query}"
        )
        return self._chat_json(prompt, task="planning")

    def decompose_query(self, query: str, max_questions: int = 5) -> List[Dict[str, Any]]:
        if self.provider == "mock":
            return self._default_sub_questions(query=query, max_questions=max_questions)

        prompt = (
            f"Break this query into 3-{max_questions} focused sub-questions. "
            "Return JSON: {\"sub_questions\": [{\"question\": str, \"priority\": int, "
            "\"answered\": false, \"parent_question\": null}]}\n"
            f"Query: {query}"
        )
        data = self._chat_json(prompt, task="planning")
        candidate = data.get("sub_questions", [])
        normalized = self._normalize_sub_questions(candidate, max_questions=max_questions)
        if normalized:
            return normalized
        return self._default_sub_questions(query=query, max_questions=max_questions)

    def extract_key_info(self, sub_question: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.provider == "mock":
            claims: List[str] = []
            citations: List[str] = []
            for doc in docs[:3]:
                claims.append(f"{sub_question}: {doc['content'][:120]}")
                citations.append(doc["source_url"])
            return {"key_information": claims, "citations": citations, "contradictions": []}

        prompt = (
            "Extract key factual claims and contradictions from documents.\n"
            f"Sub-question: {sub_question}\n"
            f"Documents: {json.dumps(docs)[:6000]}\n"
            "Return JSON with key_information (list[str]), citations (list[str]), contradictions (list[str])."
        )
        return self._chat_json(prompt, task="analysis")

    def synthesize_section(self, title: str, analyzed_content: List[Dict[str, Any]]) -> str:
        if self.provider == "mock":
            lines = [f"### {title}", ""]
            for item in analyzed_content:
                for fact in item["key_information"][:2]:
                    lines.append(f"- {fact}")
            return "\n".join(lines)

        prompt = (
            f"Write a concise report section titled '{title}' using the analyzed content.\n"
            f"Content: {json.dumps(analyzed_content)[:7000]}"
        )
        return self._chat_text(prompt, task="writing")

    def summarize_report(self, query: str, sections: Dict[str, str]) -> str:
        if self.provider == "mock":
            return f"This report summarizes findings for '{query}' across {len(sections)} sections."
        prompt = (
            f"Create a short executive summary for query: {query}\n"
            f"Sections: {json.dumps(sections)[:6000]}"
        )
        return self._chat_text(prompt, task="writing")

    def _chat_text(self, prompt: str, task: str = "default") -> str:
        data = self._chat_completion(prompt, model_override=self._model_for(task), task=task)
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        return content

    def _chat_json(self, prompt: str, task: str = "default") -> Dict[str, Any]:
        text = self._chat_text(prompt, task=task)
        parsed = self._parse_json_from_text(text)
        if parsed is not None:
            return parsed

        repaired_text = self._repair_json_with_llm(text=text, task=task, strict=False)
        repaired = self._parse_json_from_text(repaired_text)
        if repaired is not None:
            return repaired

        repaired_text_strict = self._repair_json_with_llm(text=repaired_text, task=task, strict=True)
        repaired_strict = self._parse_json_from_text(repaired_text_strict)
        if repaired_strict is not None:
            return repaired_strict
        raise ValueError("LLM did not return parseable JSON.")

    def _chat_completion(
        self,
        prompt: str,
        model_override: str | None = None,
        task: str = "default",
    ) -> Dict[str, Any]:
        selected_model = model_override or self.model
        selected_provider = self.provider.lower()
        routes = self._completion_routes(selected_provider=selected_provider, selected_model=selected_model)
        previous_exc: Exception | None = None
        for idx, (provider, model) in enumerate(routes):
            route_start = time.perf_counter()
            fallback_used = idx > 0
            try:
                result = self._dispatch_completion(
                    prompt,
                    provider_override=provider,
                    model_override=model,
                )
                usage = self._extract_usage(result)
                err = ""
                if previous_exc is not None:
                    err = f"prior_failed:{type(previous_exc).__name__}"
                self._record_metric(
                    task=task,
                    model=model,
                    provider=provider,
                    success=True,
                    fallback_used=fallback_used,
                    latency_ms=(time.perf_counter() - route_start) * 1000,
                    error=err,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                    total_cost_usd=usage["total_cost_usd"],
                )
                return result
            except Exception as exc:
                self._record_metric(
                    task=task,
                    model=model,
                    provider=provider,
                    success=False,
                    fallback_used=fallback_used,
                    latency_ms=(time.perf_counter() - route_start) * 1000,
                    error=str(exc),
                )
                previous_exc = exc
                if idx >= len(routes) - 1:
                    raise
        if previous_exc is not None:
            raise previous_exc
        raise RuntimeError("No completion route available.")

    def _dispatch_completion(
        self,
        prompt: str,
        provider_override: str,
        model_override: str | None = None,
    ) -> Dict[str, Any]:
        provider = provider_override.lower()
        if provider in {"xai", "grok"}:
            return self._xai_chat_completion(prompt, model_override=model_override)
        if provider == "groq":
            return self._groq_chat_completion(prompt, model_override=model_override)
        if provider in {"openrouter", "or"}:
            return self._openrouter_chat_completion(prompt, model_override=model_override)
        if provider == "ollama":
            return self._ollama_chat_completion(prompt, model_override=model_override)
        if provider == "openai":
            return self._openai_chat_completion(prompt, model_override=model_override)
        raise ValueError(f"Unsupported LLM provider: {provider_override}")

    def _openai_chat_completion(self, prompt: str, model_override: str | None = None) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for openai provider.")
        return self._post_with_retry(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload={
                "model": model_override or self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=60,
            retry_statuses={429, 500, 502, 503, 504},
        )

    def _xai_chat_completion(self, prompt: str, model_override: str | None = None) -> Dict[str, Any]:
        token = self.xai_api_key or self.api_key
        if not token:
            raise ValueError("XAI_API_KEY is required for xai/grok provider.")
        return self._post_with_retry(
            url="https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            payload={
                "model": model_override or self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=60,
            retry_statuses={429, 500, 502, 503, 504},
        )

    def _openrouter_chat_completion(self, prompt: str, model_override: str | None = None) -> Dict[str, Any]:
        token = self.openrouter_api_key or self.api_key
        if not token:
            raise ValueError("OPENROUTER_API_KEY is required for openrouter provider.")
        return self._post_with_retry(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            payload={
                "model": model_override or self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=60,
            retry_statuses={429, 500, 502, 503, 504},
        )

    def _groq_chat_completion(self, prompt: str, model_override: str | None = None) -> Dict[str, Any]:
        token = self.groq_api_key or self.api_key
        if not token:
            raise ValueError("GROQ_API_KEY is required for groq provider.")
        return self._post_with_retry(
            url="https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            payload={
                "model": model_override or self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=60,
            retry_statuses={429, 500, 502, 503, 504},
        )

    def _ollama_chat_completion(self, prompt: str, model_override: str | None = None) -> Dict[str, Any]:
        result = self._post_with_retry(
            url=f"{self.ollama_base_url}/api/chat",
            headers={"Content-Type": "application/json"},
            payload={
                "model": model_override or self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=120,
            retry_statuses={429, 500, 502, 503, 504},
        )
        message = result.get("message", {}) if isinstance(result, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        usage = {
            "prompt_tokens": int(result.get("prompt_eval_count", 0) or 0),
            "completion_tokens": int(result.get("eval_count", 0) or 0),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return {
            "choices": [{"message": {"content": content}}],
            "usage": usage,
        }

    def _model_for(self, task: str) -> str:
        specific = self.task_models.get(task, "").strip()
        return specific or self.model

    def _fallback_model_for(self, selected_model: str) -> str:
        candidate = (self.fallback_model or self.model).strip()
        if not candidate or candidate == selected_model:
            return ""
        return candidate

    def _fallback_provider_for(self, selected_provider: str) -> str:
        candidate = (self.fallback_provider or selected_provider).strip().lower()
        return candidate

    def _completion_routes(self, selected_provider: str, selected_model: str) -> List[tuple[str, str]]:
        routes: List[tuple[str, str]] = [(selected_provider, selected_model)]
        if not self.fallback_enabled:
            return routes

        first_provider = self._fallback_provider_for(selected_provider)
        first_model = self._fallback_model_for(selected_model)
        self._append_unique_route(routes, first_provider, first_model)

        second_provider = (self.second_fallback_provider or "").strip().lower()
        second_model = (self.second_fallback_model or "").strip()
        self._append_unique_route(routes, second_provider, second_model)
        return routes

    def _append_unique_route(self, routes: List[tuple[str, str]], provider: str, model: str) -> None:
        if not provider or not model:
            return
        key = (provider.strip().lower(), model.strip())
        if key not in routes:
            routes.append(key)

    def _default_sub_questions(self, query: str, max_questions: int) -> List[Dict[str, Any]]:
        seed = [
            "Core concepts and definitions",
            "Current adoption and trends",
            "Benefits and opportunities",
            "Risks and limitations",
            "Future outlook",
        ]
        items: List[Dict[str, Any]] = []
        for i, label in enumerate(seed[:max_questions], start=1):
            items.append({"question": f"{label} of {query}", "priority": i, "answered": False, "parent_question": None})
        return items[: max(3, min(max_questions, len(items)))]

    def _normalize_sub_questions(self, candidate: Any, max_questions: int) -> List[Dict[str, Any]]:
        if not isinstance(candidate, list):
            return []
        out: List[Dict[str, Any]] = []
        for idx, row in enumerate(candidate, start=1):
            if not isinstance(row, dict):
                continue
            question = row.get("question", "")
            if not isinstance(question, str) or not question.strip():
                continue
            priority = row.get("priority", idx)
            try:
                priority_int = int(priority)
            except (TypeError, ValueError):
                priority_int = idx
            out.append(
                {
                    "question": question.strip(),
                    "priority": priority_int,
                    "answered": bool(row.get("answered", False)),
                    "parent_question": row.get("parent_question"),
                }
            )
            if len(out) >= max_questions:
                break
        return out

    def _record_metric(
        self,
        task: str,
        model: str,
        provider: str,
        success: bool,
        fallback_used: bool,
        latency_ms: float,
        error: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        total_cost_usd: float = 0.0,
    ) -> None:
        key = f"{task}:{provider}:{model}"
        current = self.metrics.get(
            key,
            {
                "task": task,
                "provider": provider,
                "model": model,
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "fallback_calls": 0,
                "total_latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "last_error": "",
            },
        )
        current["calls"] += 1
        current["total_latency_ms"] += latency_ms
        current["prompt_tokens"] += prompt_tokens
        current["completion_tokens"] += completion_tokens
        current["total_tokens"] += total_tokens
        current["total_cost_usd"] += total_cost_usd
        if success:
            current["successes"] += 1
        else:
            current["failures"] += 1
        if fallback_used:
            current["fallback_calls"] += 1
        if error:
            current["last_error"] = error
        self.metrics[key] = current

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        return {k: dict(v) for k, v in self.metrics.items()}

    def reset_metrics(self) -> None:
        self.metrics = {}

    def _extract_usage(self, result: Dict[str, Any]) -> Dict[str, Any]:
        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        prompt_tokens = int(float(usage.get("prompt_tokens", 0) or 0))
        completion_tokens = int(float(usage.get("completion_tokens", 0) or 0))
        total_tokens = int(float(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0))

        raw_cost = 0.0
        if "total_cost" in usage:
            raw_cost = float(usage.get("total_cost", 0.0) or 0.0)
        elif "cost" in usage:
            raw_cost = float(usage.get("cost", 0.0) or 0.0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_cost_usd": raw_cost,
        }

    def _parse_json_from_text(self, text: str) -> Dict[str, Any] | None:
        candidates = self._json_candidates(text)
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and self._is_json_compatible(parsed):
                    return parsed
            except json.JSONDecodeError:
                pass
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict) and self._is_json_compatible(parsed):
                    return parsed
            except (ValueError, SyntaxError):
                continue
        return None

    def _json_candidates(self, text: str) -> List[str]:
        s = (text or "").strip()
        out: List[str] = []

        fenced = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, flags=re.IGNORECASE)
        out.extend(fenced)

        out.append(s.strip("`"))

        balanced = self._extract_balanced_object(s)
        if balanced:
            out.append(balanced)

        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            out.append(s[start : end + 1])

        # Deduplicate while preserving order.
        seen = set()
        uniq = []
        for item in out:
            key = item.strip()
            if key and key not in seen:
                seen.add(key)
                uniq.append(key)
        return uniq

    def _extract_balanced_object(self, text: str) -> str:
        start = text.find("{")
        if start == -1:
            return ""
        depth = 0
        in_str = False
        esc = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return ""

    def _is_json_compatible(self, value: Any) -> bool:
        if value is None or isinstance(value, (str, int, float, bool)):
            return True
        if isinstance(value, list):
            return all(self._is_json_compatible(item) for item in value)
        if isinstance(value, dict):
            return all(isinstance(k, str) and self._is_json_compatible(v) for k, v in value.items())
        return False

    def _repair_json_with_llm(self, text: str, task: str, strict: bool = False) -> str:
        if strict:
            prompt = (
                "You are a JSON sanitizer. Output ONLY one strict JSON object. "
                "Use double quotes for all keys/strings. No comments, no markdown, no prose.\n\n"
                f"CONTENT:\n{text[:10000]}"
            )
        else:
            prompt = (
                "Convert the following content into strict valid JSON object only. "
                "No markdown, no prose, no code fences. Return JSON object.\n\n"
                f"CONTENT:\n{text[:10000]}"
            )
        return self._chat_text(prompt, task=task)

    def _post_with_retry(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout: int,
        retry_statuses: set[int],
    ) -> Dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.retry_max_attempts):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if not self._should_retry(exc, status, retry_statuses) or attempt >= self.retry_max_attempts - 1:
                    raise
                sleep_s = self._retry_delay_seconds(exc=exc, attempt=attempt)
                time.sleep(sleep_s)
        if last_exc:
            raise last_exc
        raise RuntimeError("Request failed without exception details.")

    def _should_retry(self, exc: requests.RequestException, status: int | None, retry_statuses: set[int]) -> bool:
        if status is None:
            return isinstance(exc, (requests.Timeout, requests.ConnectionError))
        return status in retry_statuses

    def _retry_delay_seconds(self, exc: requests.RequestException, attempt: int) -> float:
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            retry_after = response.headers.get("Retry-After", "")
            try:
                retry_after_s = float(retry_after)
                if retry_after_s >= 0:
                    return min(self.retry_max_delay_s, retry_after_s)
            except (TypeError, ValueError):
                pass
        exp = self.retry_base_delay_s * (2 ** attempt)
        jitter = random.uniform(0, min(0.5, self.retry_base_delay_s + 0.1))
        return min(self.retry_max_delay_s, exp + jitter)
