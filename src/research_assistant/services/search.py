from datetime import datetime, timezone
from typing import Dict, List

import requests


class SearchService:
    def __init__(self, provider: str = "mock", api_key: str = "", serpapi_api_key: str = ""):
        self.provider = provider
        self.api_key = api_key
        self.serpapi_api_key = serpapi_api_key

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        if self.provider == "mock":
            now = datetime.now(timezone.utc)
            return [
                {
                    "title": f"Mock source {i+1} for {query}",
                    "url": f"mock://{query.replace(' ', '_')}/{i+1}",
                    "snippet": f"Snippet {i+1} discussing {query}.",
                    "source": "mock",
                    "timestamp": now,
                    "relevance_score": 0.9 - (i * 0.05),
                }
                for i in range(max_results)
            ]

        if self.provider == "serpapi":
            return self._search_serpapi(query=query, max_results=max_results)

        if self.provider != "tavily":
            raise ValueError(f"Unsupported search provider: {self.provider}")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is required for tavily provider.")

        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        now = datetime.now(timezone.utc)
        items = []
        for row in data.get("results", []):
            items.append(
                {
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "snippet": row.get("content", ""),
                    "source": "tavily",
                    "timestamp": now,
                    "relevance_score": 0.8,
                }
            )
        return items

    def _search_serpapi(self, query: str, max_results: int = 5) -> List[Dict]:
        if not self.serpapi_api_key:
            raise ValueError("SERPAPI_API_KEY is required for serpapi provider.")
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "q": query,
                "engine": "google",
                "api_key": self.serpapi_api_key,
                "num": max_results,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        now = datetime.now(timezone.utc)
        items = []
        for row in data.get("organic_results", [])[:max_results]:
            items.append(
                {
                    "title": row.get("title", ""),
                    "url": row.get("link", ""),
                    "snippet": row.get("snippet", ""),
                    "source": "serpapi",
                    "timestamp": now,
                    "relevance_score": 0.8,
                }
            )
        return items

    def fetch_url(self, url: str):
        if url.startswith("mock://"):
            topic = url.replace("mock://", "").replace("_", " ")
            return (
                f"Article for {topic}. It includes evidence, examples, and limitations. "
                "This source provides factual context for synthesis."
            )
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            return response.content
        return response.text
