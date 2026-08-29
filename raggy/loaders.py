import logging
from pathlib import Path

from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

DEFAULT_OCR_DPI = 150

_ocr_engine = None


def _get_ocr_engine():
    """Return a lazily-initialized shared RapidOCR engine instance."""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    return _ocr_engine


def _ocr_image_bytes(image_bytes: bytes) -> str:
    """Run OCR on raw image bytes and return the recognized text."""
    result, _ = _get_ocr_engine()(image_bytes)
    if not result:
        return ""
    return "\n".join(line[1] for line in result if line[1])


def _load_image(path: Path) -> list[Document]:
    """OCR a single image file into one Document."""
    text = _ocr_image_bytes(path.read_bytes())
    if not text.strip():
        logger.warning("OCR produced no text for '%s'.", path)
    return [Document(page_content=text, metadata={"source": str(path)})]


def _load_ocr_pdf(path: Path) -> list[Document]:
    """OCR an image-only PDF by rendering each page and running OCR per page."""
    import pymupdf

    documents: list[Document] = []
    with pymupdf.open(str(path)) as pdf:
        for page_number in range(pdf.page_count):
            page = pdf[page_number]
            pixmap = page.get_pixmap(dpi=DEFAULT_OCR_DPI)
            text = _ocr_image_bytes(pixmap.tobytes("png"))
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "page": page_number + 1},
                )
            )
    return documents


def _collect_text_frame(frame, text_parts: list[str]) -> None:
    """Append the text of every non-empty paragraph in a text frame."""
    text = "\n".join(
        paragraph.text for paragraph in frame.paragraphs if paragraph.text.strip()
    )
    if text.strip():
        text_parts.append(text)


def _collect_shape(shape, text_parts: list[str]) -> None:
    """Recursively collect text from a shape, its text frames, and its tables."""
    if getattr(shape, "has_text_frame", False):
        _collect_text_frame(shape.text_frame, text_parts)
    elif getattr(shape, "has_table", False):
        for row in shape.table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                text_parts.append(" | ".join(cells))
    elif getattr(shape, "shape_type", None) == 6:  # MSO_SHAPE_TYPE.GROUP
        for child in shape.shapes:
            _collect_shape(child, text_parts)


def _load_pptx(path: Path) -> list[Document]:
    """Extract text from every slide of a PowerPoint deck, one Document per slide."""
    from pptx import Presentation

    presentation = Presentation(str(path))
    documents: list[Document] = []

    for index, slide in enumerate(presentation.slides, start=1):
        text_parts: list[str] = []

        for shape in slide.shapes:
            _collect_shape(shape, text_parts)

        if text_parts:
            documents.append(
                Document(
                    page_content="\n\n".join(text_parts),
                    metadata={"source": str(path), "page": index},
                )
            )

    return documents


def _load_file(path: Path, skipped: list[Path] | None = None) -> list[Document]:
    """Load documents from a single supported file based on its extension.

    Unsupported file types are ignored and recorded in ``skipped`` instead of
    raising an error.
    """
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        if skipped is not None:
            skipped.append(path)
        return []

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        documents = loader.load()
        for doc in documents:
            if "page" in doc.metadata:
                doc.metadata["page"] += 1
        if not any((doc.page_content or "").strip() for doc in documents):
            logger.info("No extractable text in '%s'; falling back to OCR...", path)
            documents = _load_ocr_pdf(path)
        return documents

    if suffix in IMAGE_EXTENSIONS:
        return _load_image(path)

    if suffix == ".docx":
        return Docx2txtLoader(str(path)).load()

    if suffix == ".pptx":
        return _load_pptx(path)

    if suffix in {".html", ".htm"}:
        return BSHTMLLoader(str(path), open_encoding="utf-8").load()

    loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()


def _load_directory(path: Path, skipped: list[Path] | None = None) -> list[Document]:
    """Recursively load documents from every supported file inside a directory."""
    found = False
    documents: list[Document] = []

    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            if skipped is not None:
                skipped.append(file_path)
            continue

        found = True
        try:
            documents.extend(_load_file(file_path))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load '%s': %s", file_path, e)

    if not found:
        raise FileNotFoundError(
            f"No supported files (txt/md/pdf/docx/pptx/html/img) found under "
            f"directory: {path}"
        )

    return documents


def _report_unsupported(skipped: list[Path]) -> None:
    """Log a single info message summarizing ignored unsupported files."""
    if not skipped:
        return
    logger.info(
        "Detected %d unsupported file(s) that won't be used; ignoring them. "
        "Supported file types: %s.",
        len(skipped),
        ", ".join(sorted(SUPPORTED_EXTENSIONS)),
    )


def load_documents(source: str, skipped: list[Path] | None = None) -> list[Document]:
    """Load documents from a single file or a directory containing supported files.

    Unsupported file types are ignored rather than raising an error. When
    ``skipped`` is not provided, a single info message about any ignored files
    is logged.
    """
    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"Source document not found at: {source}")

    owns_skipped = skipped is None
    if owns_skipped:
        skipped = []

    if path.is_dir():
        logger.info("Loading documents from directory '%s'...", source)
        documents = _load_directory(path, skipped)
    else:
        logger.info("Loading document from '%s'...", source)
        documents = _load_file(path, skipped)

    if owns_skipped:
        _report_unsupported(skipped)
    return documents


def load_documents_from_sources(sources: list[str]) -> list[Document]:
    """Load documents from multiple files and/or directories.

    Each entry in ``sources`` may be a single supported file or a directory
    containing supported files. Any missing source raises ``FileNotFoundError``
    before any documents are loaded. Unsupported file types are ignored and
    reported once via a single info message.
    """
    paths = [Path(source) for source in sources]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Source document(s) not found at: " + ", ".join(missing)
        )

    documents: list[Document] = []
    skipped: list[Path] = []
    for path in paths:
        documents.extend(load_documents(str(path), skipped))
    _report_unsupported(skipped)
    return documents
