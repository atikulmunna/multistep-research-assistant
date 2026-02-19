from io import BytesIO

from pypdf import PdfWriter

from research_assistant.services.parser import DocumentParser


def _make_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.add_metadata({"/Title": "Test PDF"})
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_detect_pdf_by_header():
    parser = DocumentParser()
    data = _make_pdf_bytes()
    assert parser.detect_type("https://example.com/file", data) == "pdf"


def test_parse_pdf_returns_string_content():
    parser = DocumentParser()
    data = _make_pdf_bytes()
    doc = parser.parse(data, "pdf", "https://example.com/sample.pdf")
    assert doc["doc_type"] == "pdf"
    assert isinstance(doc["content"], str)


def test_pdf_url_with_non_pdf_content_falls_back_to_text():
    parser = DocumentParser()
    content = "<html><body>Not a pdf</body></html>"
    doc_type = parser.detect_type("https://example.com/fake.pdf", content)
    assert doc_type == "html"
    doc = parser.parse(content, doc_type, "https://example.com/fake.pdf")
    assert doc["doc_type"] == "html"
    assert "Not a pdf" in doc["content"]
