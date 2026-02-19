from research_assistant.main import _markdown_to_html


def test_markdown_to_html_renders_headings_lists_and_citations():
    markdown = "\n".join(
        [
            "# Report Title",
            "",
            "- first item",
            "- second item",
            "",
            "[1] [https://example.com](https://example.com)",
        ]
    )
    body = _markdown_to_html(markdown, "Research Report")
    assert "<h1>Research Report</h1>" in body
    assert "<ul>" in body
    assert "<li>first item</li>" in body
    assert "class='citation'" in body
    assert '<a href="https://example.com"' in body


def test_markdown_to_html_autolinks_raw_urls():
    body = _markdown_to_html("See https://example.org/docs for details.", "t")
    assert '<a href="https://example.org/docs"' in body
