from io import BytesIO
from typing import Dict, Union

from bs4 import BeautifulSoup
from pypdf import PdfReader


class DocumentParser:
    def detect_type(self, url: str, content: Union[str, bytes]) -> str:
        lower = url.lower()
        if isinstance(content, bytes):
            if content.startswith(b"%PDF"):
                return "pdf"
            if lower.endswith(".md"):
                return "md"
            sample = content[:2048].decode("utf-8", errors="ignore").lower()
            if "<html" in sample:
                return "html"
            return "txt"
        if lower.endswith(".pdf") and self._looks_like_pdf_text(content):
            return "pdf"
        if lower.endswith(".md"):
            return "md"
        if "<html" in content.lower():
            return "html"
        return "txt"

    def parse(self, content: Union[str, bytes], doc_type: str, source_url: str) -> Dict:
        if doc_type == "html":
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(" ", strip=True)
            title = (soup.title.string if soup.title and soup.title.string else source_url).strip()
            return {"content": text, "metadata": {"title": title}, "source_url": source_url, "doc_type": "html"}

        if doc_type == "pdf":
            text = self._parse_pdf(content)
            return {
                "content": text[:15000],
                "metadata": {"title": source_url},
                "source_url": source_url,
                "doc_type": "pdf",
            }

        if doc_type in {"txt", "md"}:
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            return {
                "content": content[:15000],
                "metadata": {"title": source_url},
                "source_url": source_url,
                "doc_type": doc_type,
            }

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        return {"content": content[:15000], "metadata": {}, "source_url": source_url, "doc_type": "txt"}

    def _parse_pdf(self, content: Union[str, bytes]) -> str:
        if isinstance(content, str):
            data = content.encode("utf-8", errors="ignore")
        else:
            data = content
        try:
            reader = PdfReader(BytesIO(data))
            chunks = []
            for page in reader.pages:
                chunks.append(page.extract_text() or "")
            return "\n".join(chunks).strip()
        except Exception:
            return data.decode("utf-8", errors="ignore").strip()

    def _looks_like_pdf_text(self, content: str) -> bool:
        sample = content[:32].lstrip()
        return sample.startswith("%PDF")
