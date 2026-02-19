from typing import Dict, List


class CitationManager:
    def __init__(self) -> None:
        self._map: Dict[str, int] = {}
        self._ordered: List[str] = []

    def add_source(self, url: str) -> str:
        if url not in self._map:
            self._map[url] = len(self._ordered) + 1
            self._ordered.append(url)
        return f"[{self._map[url]}]"

    def bibliography(self) -> List[str]:
        out: List[str] = []
        for idx, url in enumerate(self._ordered, start=1):
            if isinstance(url, str) and "://" in url:
                out.append(f"[{idx}] [{url}]({url})")
            else:
                out.append(f"[{idx}] {url}")
        return out
