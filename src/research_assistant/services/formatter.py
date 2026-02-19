from datetime import datetime, timezone
from typing import Dict, List


class ReportFormatter:
    def to_markdown(
        self,
        query: str,
        summary: str,
        sections: Dict[str, str],
        key_findings: List[str],
        gaps: List[str],
        references: List[str],
    ) -> str:
        parts = [
            f"# Research Report: {query}",
            "",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "",
            "## Executive Summary",
            summary,
            "",
        ]

        for idx, (title, content) in enumerate(sections.items(), start=1):
            parts.extend([f"## {idx}. {title}", content, ""])

        parts.append("## Key Findings")
        if key_findings:
            parts.extend([f"- {item}" for item in key_findings])
        else:
            parts.append("- No high-confidence findings were extracted.")
        parts.append("")

        parts.append("## Areas for Further Research")
        if gaps:
            parts.extend([f"- {item}" for item in gaps])
        else:
            parts.append("- No critical gaps identified.")
        parts.append("")

        parts.append("## References")
        parts.extend(references if references else ["- No references collected."])
        parts.append("")
        return "\n".join(parts)
