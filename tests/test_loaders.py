import logging

import pytest

from raggy import loaders
from raggy.loaders import load_documents, load_documents_from_sources


def test_load_documents_raises_for_missing_source():
    with pytest.raises(FileNotFoundError):
        load_documents("/nonexistent/path.txt")


def test_load_documents_from_sources_loads_multiple_sources(tmp_path):
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    (dir_a / "one.txt").write_text("Text A", encoding="utf-8")
    (tmp_path / "two.md").write_text("Text B", encoding="utf-8")

    documents = load_documents_from_sources([str(dir_a), str(tmp_path / "two.md")])

    contents = {doc.page_content for doc in documents}
    assert contents == {"Text A", "Text B"}


def test_load_documents_from_sources_raises_for_missing_source(tmp_path):
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    (dir_a / "one.txt").write_text("Text A", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="not found"):
        load_documents_from_sources([str(dir_a), str(tmp_path / "missing.txt")])


def test_load_documents_ignores_unsupported_extension(tmp_path):
    unsupported = tmp_path / "notes.csv"
    unsupported.write_text("hello", encoding="utf-8")

    documents = load_documents(str(unsupported))

    assert documents == []


def test_load_documents_reports_unsupported_file_once(tmp_path, caplog):
    (tmp_path / "a.txt").write_text("Text A", encoding="utf-8")
    (tmp_path / "notes.csv").write_text("ignored", encoding="utf-8")
    (tmp_path / "data.json").write_text("{}", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="raggy.loaders"):
        documents = load_documents_from_sources([str(tmp_path)])

    contents = {doc.page_content for doc in documents}
    assert contents == {"Text A"}

    unsupported_messages = [
        record.getMessage()
        for record in caplog.records
        if "unsupported" in record.getMessage().lower()
    ]
    assert len(unsupported_messages) == 1
    assert "2" in unsupported_messages[0]


def test_load_documents_loads_markdown_file(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("# Heading\n\nSome markdown body.", encoding="utf-8")

    documents = load_documents(str(doc))

    assert len(documents) == 1
    assert "Some markdown body." in documents[0].page_content


def test_load_documents_loads_text_file(tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("Hello world", encoding="utf-8")

    documents = load_documents(str(doc))

    assert len(documents) == 1
    assert documents[0].page_content == "Hello world"


def test_load_documents_loads_pdf_file(tmp_path):
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "sample.pdf"
    pdf = canvas.Canvas(str(pdf_path))
    pdf.drawString(100, 750, "PDF content here")
    pdf.save()

    documents = load_documents(str(pdf_path))

    assert len(documents) == 1
    assert "PDF content here" in documents[0].page_content
    assert documents[0].metadata["page"] == 1


def test_load_documents_pdf_pages_are_one_indexed(tmp_path):
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "pages.pdf"
    pdf = canvas.Canvas(str(pdf_path))
    for _ in range(3):
        pdf.drawString(100, 750, "Page content")
        pdf.showPage()
    pdf.save()

    documents = load_documents(str(pdf_path))

    assert [doc.metadata["page"] for doc in documents] == [1, 2, 3]


def test_load_documents_recursively_loads_directory(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_text("Text A", encoding="utf-8")
    (tmp_path / "nested" / "b.txt").write_text("Text B", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("ignored", encoding="utf-8")

    documents = load_documents(str(tmp_path))

    contents = {doc.page_content for doc in documents}
    assert contents == {"Text A", "Text B"}


def test_load_documents_raises_when_directory_has_no_supported_files(tmp_path):
    (tmp_path / "notes.csv").write_text("ignored", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No supported files"):
        load_documents(str(tmp_path))


def test_supported_extensions_exported():
    assert loaders.SUPPORTED_EXTENSIONS == {
        ".txt",
        ".md",
        ".markdown",
        ".pdf",
        ".docx",
        ".pptx",
        ".html",
        ".htm",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
    }


def test_load_documents_loads_docx_file(tmp_path):
    from docx import Document as DocxDocument

    docx_path = tmp_path / "notes.docx"
    docx = DocxDocument()
    docx.add_paragraph("Hello from docx")
    docx.save(str(docx_path))

    documents = load_documents(str(docx_path))

    assert len(documents) == 1
    assert "Hello from docx" in documents[0].page_content
    assert "page" not in documents[0].metadata


def test_load_documents_loads_pptx_file(tmp_path):
    from pptx import Presentation

    pptx_path = tmp_path / "deck.pptx"
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "Slide Title"
    presentation.save(str(pptx_path))

    documents = load_documents(str(pptx_path))

    assert len(documents) == 1
    assert "Slide Title" in documents[0].page_content
    assert documents[0].metadata["page"] == 1


def test_load_documents_loads_html_file(tmp_path):
    html_path = tmp_path / "page.html"
    html_path.write_text(
        "<html><body><h1>Heading</h1><p>Some HTML body.</p></body></html>",
        encoding="utf-8",
    )

    documents = load_documents(str(html_path))

    assert len(documents) == 1
    assert "Some HTML body." in documents[0].page_content


def test_load_documents_directory_includes_new_formats(tmp_path):
    from docx import Document as DocxDocument

    docx = DocxDocument()
    docx.add_paragraph("Docx content")
    docx.save(str(tmp_path / "report.docx"))
    (tmp_path / "page.html").write_text(
        "<html><body>HTML content</body></html>", encoding="utf-8"
    )

    documents = load_documents(str(tmp_path))

    contents = " ".join(doc.page_content for doc in documents)
    assert "Docx content" in contents
    assert "HTML content" in contents


def _make_ocr_image(path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, 48)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    img = Image.new("RGB", (1000, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "HELLO WORLD", fill="black", font=font)
    draw.text((20, 110), "RAGGY PIPELINE", fill="black", font=font)
    img.save(str(path))


def _compact(text: str) -> str:
    return "".join(text.split())


def test_load_documents_loads_image_file(tmp_path):
    image_path = tmp_path / "scan.png"
    _make_ocr_image(image_path)

    documents = load_documents(str(image_path))

    assert len(documents) == 1
    assert "HELLOWORLD" in _compact(documents[0].page_content)


def test_load_documents_ocr_fallback_for_scanned_pdf(tmp_path):
    import pymupdf

    image_path = tmp_path / "scan.png"
    _make_ocr_image(image_path)

    pdf_path = tmp_path / "scanned.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=1000, height=300)
    page.insert_image(pymupdf.Rect(0, 0, 1000, 300), filename=str(image_path))
    doc.save(str(pdf_path))

    documents = load_documents(str(pdf_path))

    assert len(documents) == 1
    assert "HELLOWORLD" in _compact(documents[0].page_content)
    assert documents[0].metadata["page"] == 1
