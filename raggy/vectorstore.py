import hashlib
import logging
import math
import shutil
import sys
from pathlib import Path

import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .loaders import SUPPORTED_EXTENSIONS, load_documents_from_sources

logger = logging.getLogger(__name__)


def annotate_line_numbers(splits: list[Document], content: str) -> None:
    """Annotate text splits with ``start_line``/``end_line`` line ranges.

    The positions are derived from where each chunk's text appears in the
    ``content`` string (1-indexed line numbers). Because ``chunk_overlap``
    causes adjacent chunks to share text, a search from ``cursor`` may not
    find a chunk against its true start; the first-occurrence fallback still
    yields approximate (but useful) line attribution.
    """
    cursor = 0
    chunk_size_tolerance = max(0, len(content))
    for split in splits:
        text = split.page_content
        start = content.find(text, cursor)
        if start == -1:
            start = content.find(text, 0, cursor + chunk_size_tolerance)
        if start == -1:
            start = content.find(text)

        split.metadata["start_line"] = content.count("\n", 0, start) + 1
        if text.endswith("\n"):
            newlines_in_text = text.count("\n") - 1
        else:
            newlines_in_text = text.count("\n")
        split.metadata["end_line"] = split.metadata["start_line"] + newlines_in_text
        cursor = start + len(text)


_LINE_ANNOTATED_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm"}


def _should_annotate_lines(doc: Document) -> bool:
    """Return True only for text-based files that get line-number annotations.

    PDFs and PowerPoint decks carry a ``page`` key set by their loaders. DOCX
    has no native page boundaries (and Word's pagination can't be reproduced
    reliably), and image files have no meaningful lines, so both carry no
    location metadata and are skipped here.
    """
    suffix = Path(doc.metadata.get("source", "")).suffix.lower()
    return suffix in _LINE_ANNOTATED_EXTENSIONS


def get_embeddings(model_name: str) -> OllamaEmbeddings:
    """Initialize and return OllamaEmbeddings."""
    return OllamaEmbeddings(model=model_name)


def get_vectorstore(persist_directory: str, embedding_model: str) -> Chroma:
    """Initialize and return the Chroma vector store."""
    embeddings = get_embeddings(embedding_model)
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def close_vectorstore(vectorstore: Chroma) -> None:
    """Release the client's underlying DB connection.

    Chroma holds an open handle to ``chroma.sqlite3``. Deleting the persist
    dir while that handle is open leaves a stale connection to a removed
    inode, and the next write fails with "attempt to write a readonly
    database". Call this before wiping/rebuilding the DB from source.
    """
    client = getattr(vectorstore, "_client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()


def ingest_document(
    sources: list[str],
    vectorstore: Chroma,  # Assuming Chroma is imported in your actual file
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
) -> None:
    """Loads, splits, and embeds documents into the vector store in batches.

    Each entry in ``sources`` may point to a single supported file (e.g.
    .txt/.md/.pdf/.docx/.pptx/.html/.png) or to a directory containing multiple
    supported files.

    Chunks are grouped into batches of up to ``batch_size`` chunks. The number
    of batches is derived dynamically from the total chunk count, so the setting
    stays sensible regardless of dataset size.
    """
    docs = load_documents_from_sources(sources)

    if not docs:
        logger.warning("No documents found to index.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits: list[Document] = []
    for doc in docs:
        doc_splits = text_splitter.split_documents([doc])
        if _should_annotate_lines(doc):
            annotate_line_numbers(doc_splits, doc.page_content)
        splits.extend(doc_splits)
    total_splits = len(splits)
    logger.info("Split text into %d chunks.", total_splits)

    if not total_splits:
        logger.warning("No content found to split and index.")
        return

    # Group splits into batches of at most ``batch_size`` chunks. The number of
    # batches is computed from the total chunk count so it adapts to the dataset
    # size: small datasets embed in a handful of batches, large ones in many.
    n_batches = math.ceil(total_splits / batch_size)
    batch_sizes = [
        min(batch_size, total_splits - i * batch_size) for i in range(n_batches)
    ]

    logger.info(
        "Embedding chunks into Chroma (%d batches of <= %d)...",
        n_batches,
        batch_size,
    )

    for size in tqdm(
        batch_sizes,
        total=n_batches,
        desc="Indexing Batches",
        disable=not sys.stderr.isatty(),
    ):
        batch = splits[:size]
        splits = splits[size:]
        vectorstore.add_documents(batch)


_HASH_BLOCK_SIZE = 1048576


def _file_fingerprint(sources: list[str], block_size: int = _HASH_BLOCK_SIZE) -> str:
    """Return a content-based fingerprint of the source file set.

    Walks each entry in ``sources`` the same way ``load_documents`` indexes it
    and folds each supported file's (relative path, content hash) into a single
    digest. Hashing per-file (rather than a single concatenated blob) also
    catches file additions, deletions, and renames, not just content edits.
    Cheap enough to run on every pipeline start.
    """

    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                block = f.read(block_size)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    raw = []
    seen: set[Path] = set()
    for source in sources:
        root = Path(source)
        if not root.exists():
            raise FileNotFoundError(f"Source document not found at: {source}")

        if root.is_dir():
            for file_path in sorted(root.rglob("*")):
                if (
                    not file_path.is_file()
                    or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS
                ):
                    continue
                if file_path.resolve() in seen:
                    continue
                seen.add(file_path.resolve())
                try:
                    raw.append(
                        f"{file_path.relative_to(root)!s}:{_hash_file(file_path)}"
                    )
                except OSError:
                    continue
        elif root.suffix.lower() in SUPPORTED_EXTENSIONS:
            if root.resolve() in seen:
                continue
            seen.add(root.resolve())
            raw.append(f"{root.name}:{_hash_file(root)}")

    raw.sort()
    digest = hashlib.sha256()
    for entry in raw:
        digest.update(entry.encode("utf-8"))
    return digest.hexdigest()


def _load_manifest(persist_directory: str) -> dict | None:
    """Read the build manifest (if any) from the DB directory."""
    manifest_path = Path(persist_directory) / "manifest.yaml"
    if not manifest_path.exists():
        return None
    return yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}


def _write_manifest(persist_directory: str, index_cfg: dict) -> None:
    """Persist the index-affecting config so later runs can detect drift."""
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(persist_directory) / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(index_cfg), encoding="utf-8")


def _reset_persist_directory(persist_directory: str) -> None:
    """Physically delete all contents of the persist directory.

    ``reset_collection()`` only removes records but leaves stale segment /
    version files behind in the persist dir, so repeated rebuilds accumulate
    garbage. A full wipe of the directory is the clean way to rebuild.
    """
    root = Path(persist_directory)
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def build_index_config(
    sources: list[str],
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> dict:
    """Build the index-affecting config (including a fresh content fingerprint).

    The returned dict is compared against the persisted ``manifest.yaml`` to
    decide whether the DB must be rebuilt from source.
    """
    return {
        "sources": list(sources),
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "embedding_model": embedding_model,
        "content_hash": _file_fingerprint(sources),
    }


def db_needs_rebuild(persist_directory: str, index_cfg: dict) -> bool:
    """Return True if the stored manifest differs from the current index config."""
    stored_manifest = _load_manifest(persist_directory)
    return stored_manifest is None or stored_manifest != index_cfg


def initialize_db(
    persist_directory: str,
    embedding_model: str,
    sources: list[str],
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
) -> Chroma:
    """
    Initializes the Chroma database.
    If the database is empty, or the index-affecting config no longer matches
    the persisted manifest.yaml, the persist directory is wiped and the docs
    are re-indexed from scratch.
    """
    # Config that determines the index content; only these drive rebuilds.
    index_cfg = build_index_config(sources, chunk_size, chunk_overlap, embedding_model)
    config_changed = db_needs_rebuild(persist_directory, index_cfg)

    if config_changed:
        logger.info(
            "Detected configuration or source content change "
            "(sources, chunk_size, chunk_overlap, embedding_model, "
            "or docs content); rebuilding DB from source documents."
        )
        # Wipe the persist dir BEFORE opening any Chroma connection. Deleting
        # the sqlite files while a client holds an open handle leaves a stale
        # connection to a removed inode, which fails on the next write
        # ("attempt to write a readonly database").
        _reset_persist_directory(persist_directory)

    vectorstore = get_vectorstore(persist_directory, embedding_model)
    collection_count = vectorstore._collection.count()

    if collection_count == 0 or config_changed:
        ingest_document(
            sources=sources,
            vectorstore=vectorstore,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
        )
        _write_manifest(persist_directory, index_cfg)
        logger.info(
            "Successfully saved DB to '%s'.",
            persist_directory,
        )
    else:
        logger.info(
            "Loaded existing vector DB from '%s'.",
            persist_directory,
        )

    return vectorstore
