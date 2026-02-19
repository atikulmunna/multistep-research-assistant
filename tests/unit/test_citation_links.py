from research_assistant.services.citation import CitationManager
from research_assistant.main import _markdown_to_html


def test_bibliography_uses_markdown_hyperlinks_for_http_urls():
    c = CitationManager()
    c.add_source("https://example.com/a")
    c.add_source("mock://topic/1")
    rows = c.bibliography()
    assert rows[0] == "[1] [https://example.com/a](https://example.com/a)"
    assert rows[1] == "[2] [mock://topic/1](mock://topic/1)"


def test_markdown_to_html_linkifies_markdown_links():
    html = _markdown_to_html("[1] [https://example.com](https://example.com)", "t")
    assert '<a href="https://example.com"' in html
