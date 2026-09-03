import hashlib
import logging
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .bm25_retriever import save_bm25_index
from .loaders import (
    annotate_line_numbers,
    load_documents,
    should_annotate_lines,
    source_files,
)
from .progress import ProgressCallback

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.yaml"

# Manifest keys whose change invalidates every stored embedding, so the DB can
# only be rebuilt from scratch. Everything else (which files exist and what
# they contain) is reconciled file-by-file.
_REBUILD_KEYS = ("chunk_size", "chunk_overlap", "embedding_model")


@dataclass(frozen=True)
class IndexPlan:
    """What must happen to bring the DB in line with the current sources.

    ``full_rebuild`` means every stored embedding is invalid (no manifest, or a
    chunking/embedding-model change), so the persist dir is wiped and rebuilt.
    Otherwise the three file lists describe an incremental update.
    """

    full_rebuild: bool = False
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """True if the DB is out of date in any way."""
        return bool(self.full_rebuild or self.added or self.modified or self.removed)


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


def _split_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split loaded documents into overlapping chunks, annotating line numbers."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits: list[Document] = []
    for doc in docs:
        doc_splits = text_splitter.split_documents([doc])
        if should_annotate_lines(doc):
            annotate_line_numbers(doc_splits, doc.page_content)
        splits.extend(doc_splits)
    return splits


def _embed_in_batches(
    splits: list[Document],
    vectorstore: Chroma,
    batch_size: int,
    progress: ProgressCallback | None = None,
) -> None:
    """Embed ``splits`` into Chroma in batches of at most ``batch_size`` chunks.

    The number of batches is derived dynamically from the total chunk count, so
    the setting stays sensible regardless of dataset size.

    ``progress`` reports the batches on the same single status line the ingest
    step uses; the tqdm bar is suppressed then, since two live progress
    displays would fight over the same terminal line.
    """
    total_splits = len(splits)
    n_batches = math.ceil(total_splits / batch_size)
    batch_sizes = [
        min(batch_size, total_splits - i * batch_size) for i in range(n_batches)
    ]

    logger.info(
        "Embedding chunks into Chroma (%d batches of <= %d)...",
        n_batches,
        batch_size,
    )

    remaining = list(splits)
    for index, size in enumerate(
        tqdm(
            batch_sizes,
            total=n_batches,
            desc="Indexing Batches",
            disable=progress is not None or not sys.stderr.isatty(),
        ),
        start=1,
    ):
        if progress is not None:
            progress(f"[{index}/{n_batches}] embedding chunks ...")
        batch = remaining[:size]
        remaining = remaining[size:]
        vectorstore.add_documents(batch)


def create_index(
    sources: list[str],
    vectorstore: Chroma,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    persist_directory: str,
    progress: ProgressCallback | None = None,
) -> None:
    """Build the index from scratch: load, split, and embed every source file.

    The full-build counterpart to ``update_index``, which applies only the
    changed files to an index that already exists.

    Each entry in ``sources`` may point to a single supported file (e.g.
    .txt/.md/.pdf/.docx/.pptx/.html/.png) or to a directory containing multiple
    supported files.

    Also builds a BM25 index over the same chunks and persists it to
    ``<persist_directory>/bm25_index/`` for hybrid retrieval.

    ``progress`` receives one status line per file read and per embedded batch.
    """
    docs = load_documents(sources, progress=progress)

    if not docs:
        logger.warning("No documents found to index.")
        return

    splits = _split_documents(docs, chunk_size, chunk_overlap)
    logger.info("Split text into %d chunks.", len(splits))

    if not splits:
        logger.warning("No content found to split and index.")
        return

    _embed_in_batches(splits, vectorstore, batch_size, progress)
    save_bm25_index(splits, persist_directory)


# Chroma binds one SQL variable per returned row, and SQLite caps a statement
# at 32766 of them, so an unpaged get() over a large collection fails with
# "too many SQL variables". Read the corpus back one page at a time instead.
_COLLECTION_PAGE_SIZE = 10000


def _collection_chunks(vectorstore: Chroma) -> list[Document]:
    """Return every chunk currently stored in Chroma as a ``Document``.

    Reading the stored text back (rather than re-splitting the corpus) is what
    lets the BM25 index be rebuilt after an incremental update without
    touching files that did not change.
    """
    chunks: list[Document] = []
    offset = 0
    while True:
        stored = vectorstore._collection.get(
            include=["documents", "metadatas"],
            limit=_COLLECTION_PAGE_SIZE,
            offset=offset,
        )
        documents = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []
        chunks.extend(
            Document(page_content=text, metadata=dict(metadata or {}))
            for text, metadata in zip(documents, metadatas)
        )
        if len(documents) < _COLLECTION_PAGE_SIZE:
            return chunks
        offset += len(documents)


def _delete_chunks_for_files(vectorstore: Chroma, files: list[str]) -> None:
    """Delete every stored chunk whose ``source`` metadata is one of ``files``."""
    if not files:
        return
    vectorstore._collection.delete(where={"source": {"$in": files}})


def update_index(
    vectorstore: Chroma,
    plan: IndexPlan,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    persist_directory: str,
    progress: ProgressCallback | None = None,
) -> None:
    """Apply ``plan`` to an existing DB without re-embedding untouched files.

    Chunks belonging to removed or modified files are deleted from Chroma,
    then added and modified files are re-loaded, split, and embedded. The BM25
    index has no incremental update path, so it is rebuilt from the chunks now
    stored in Chroma — cheap, since that requires no embedding calls.
    """
    stale = plan.removed + plan.modified
    if stale:
        logger.info("Removing chunks for %d changed/deleted file(s).", len(stale))
        _delete_chunks_for_files(vectorstore, stale)

    reindexed = plan.added + plan.modified
    if reindexed:
        logger.info("Indexing %d new/changed file(s).", len(reindexed))
        # A file fingerprinted earlier this run may already be gone; skipping
        # it beats aborting the update, and the next run records the deletion.
        docs = load_documents(reindexed, progress=progress, on_missing="skip")
        splits = _split_documents(docs, chunk_size, chunk_overlap)
        if splits:
            _embed_in_batches(splits, vectorstore, batch_size, progress)
        else:
            logger.warning("No content found in the new/changed files.")

    save_bm25_index(_collection_chunks(vectorstore), persist_directory)


# Read in 1 MiB blocks rather than slurping whole files: the corpus can
# include large PDFs/images, and every file is hashed on each pipeline start.
# (hashlib.file_digest would replace this loop, but it needs Python 3.11.)
_HASH_BLOCK_SIZE = 1048576


def _hash_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(_HASH_BLOCK_SIZE)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def file_fingerprints(sources: list[str]) -> dict[str, str]:
    """Return a ``{file path: content hash}`` map of the source file set.

    Hashing per file (rather than folding everything into one digest) is what
    makes incremental indexing possible: comparing this map against the one in
    the manifest names exactly which files were added, modified, or deleted.
    Cheap enough to run on every pipeline start.

    The file set comes from :func:`raggy.loaders.source_files`, the same walk
    the loaders use, so these keys are exactly the ``source`` metadata values
    stored on the chunks they fingerprint.
    """
    fingerprints: dict[str, str] = {}
    for file_path in source_files(sources):
        try:
            fingerprints[str(file_path)] = _hash_file(file_path)
        except OSError:
            continue
    return fingerprints


def _load_manifest(persist_directory: str) -> dict | None:
    """Read the build manifest (if any) from the DB directory."""
    manifest_path = Path(persist_directory) / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    return yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}


def _write_manifest(persist_directory: str, index_cfg: dict) -> None:
    """Persist the index-affecting config so later runs can detect drift."""
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(persist_directory) / MANIFEST_FILENAME
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
    """Build the index-affecting config (including fresh per-file fingerprints).

    The returned dict is compared against the persisted ``manifest.yaml`` to
    decide whether the DB is stale and, if so, which files need re-indexing.
    """
    return {
        "sources": list(sources),
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "embedding_model": embedding_model,
        "files": file_fingerprints(sources),
    }


def plan_index_update(persist_directory: str, index_cfg: dict) -> IndexPlan:
    """Diff the current index config against the stored manifest.

    A missing manifest, a manifest without per-file fingerprints (written by an
    older version), or a change to chunking/embedding settings forces a full
    rebuild. Everything else is reduced to the set of files that were added,
    modified, or deleted since the last build.
    """
    stored = _load_manifest(persist_directory)
    if stored is None:
        return IndexPlan(full_rebuild=True)

    if any(stored.get(key) != index_cfg[key] for key in _REBUILD_KEYS):
        return IndexPlan(full_rebuild=True)

    stored_files = stored.get("files")
    if not isinstance(stored_files, dict):
        return IndexPlan(full_rebuild=True)

    current_files = index_cfg["files"]
    return IndexPlan(
        added=sorted(set(current_files) - set(stored_files)),
        modified=sorted(
            path
            for path, digest in current_files.items()
            if path in stored_files and stored_files[path] != digest
        ),
        removed=sorted(set(stored_files) - set(current_files)),
    )


def db_needs_rebuild(persist_directory: str, index_cfg: dict) -> bool:
    """Return True if the stored manifest no longer matches the current sources.

    Covers both kinds of staleness (full rebuild and incremental update); use
    ``plan_index_update`` when the distinction matters.
    """
    return plan_index_update(persist_directory, index_cfg).has_changes


def initialize_db(
    persist_directory: str,
    embedding_model: str,
    sources: list[str],
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    progress: ProgressCallback | None = None,
) -> Chroma:
    """
    Initializes the Chroma database.

    If the database is empty, or ``chunk_size``/``chunk_overlap``/
    ``embedding_model`` no longer match the persisted manifest.yaml, the
    persist directory is wiped and the docs are re-indexed from scratch. If
    only the source files changed, the DB is updated incrementally: just the
    added and modified files are embedded and the removed ones dropped.

    ``progress`` receives a status line for each step of that work (see
    :func:`create_index`); it is never called when the DB is already current.
    """
    # Config that determines the index content; only these drive re-indexing.
    index_cfg = build_index_config(sources, chunk_size, chunk_overlap, embedding_model)
    plan = plan_index_update(persist_directory, index_cfg)

    if plan.full_rebuild:
        logger.info(
            "Detected an index config change (chunk_size, chunk_overlap, or "
            "embedding_model); rebuilding DB from source documents."
        )
        # Wipe the persist dir BEFORE opening any Chroma connection. Deleting
        # the sqlite files while a client holds an open handle leaves a stale
        # connection to a removed inode, which fails on the next write
        # ("attempt to write a readonly database").
        _reset_persist_directory(persist_directory)

    vectorstore = get_vectorstore(persist_directory, embedding_model)
    collection_count = vectorstore._collection.count()

    if plan.full_rebuild or collection_count == 0:
        create_index(
            sources=sources,
            vectorstore=vectorstore,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            persist_directory=persist_directory,
            progress=progress,
        )
        _write_manifest(persist_directory, index_cfg)
        logger.info("Successfully saved DB to '%s'.", persist_directory)
    elif plan.has_changes:
        logger.info(
            "Detected source changes (%d added, %d modified, %d deleted); "
            "updating DB incrementally.",
            len(plan.added),
            len(plan.modified),
            len(plan.removed),
        )
        update_index(
            vectorstore=vectorstore,
            plan=plan,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            persist_directory=persist_directory,
            progress=progress,
        )
        _write_manifest(persist_directory, index_cfg)
        logger.info("Successfully updated DB in '%s'.", persist_directory)
    else:
        logger.info("Loaded existing vector DB from '%s'.", persist_directory)

    return vectorstore
