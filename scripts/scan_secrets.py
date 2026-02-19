from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "reports",
    "src/research_assistant.egg-info",
}
EXCLUDED_FILES = {
    ".env",
}
MAX_BYTES = 1_000_000

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("OpenRouter key", re.compile(r"\bsk-or-v1-[A-Za-z0-9]{20,}\b")),
    ("Groq key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("Tavily key", re.compile(r"\btvly-[A-Za-z0-9-]{16,}\b")),
    ("xAI key", re.compile(r"\bxai-[A-Za-z0-9]{16,}\b")),
    ("SerpAPI key", re.compile(r"(?i)\bserpapi_api_key\s*=\s*['\"]?[0-9a-f]{20,}")),
]


def _is_excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if path.name in EXCLUDED_FILES:
        return True
    return any(rel == d or rel.startswith(f"{d}/") for d in EXCLUDED_DIRS)


def _is_text_file(path: Path) -> bool:
    try:
        raw = path.read_bytes()
    except Exception:
        return False
    if len(raw) > MAX_BYTES:
        return False
    return b"\x00" not in raw


def main() -> int:
    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if _is_excluded(path):
            continue
        if not _is_text_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for label, pattern in PATTERNS:
            for match in pattern.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                rel = path.relative_to(ROOT).as_posix()
                hits.append(f"{rel}:{line_no}: {label}")
                break

    if hits:
        print("Potential secrets detected:")
        for item in hits:
            print(f"- {item}")
        print("\nMove secrets to .env and rotate exposed keys.")
        return 1

    print("No obvious secrets found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
