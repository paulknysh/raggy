import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from langchain_chroma import Chroma

from .config import RaggySettings
from .llm_factory import ensure_ollama_model
from .pipeline import build_rag_chain
from .vectorstore import (
    build_index_config,
    close_vectorstore,
    db_needs_rebuild,
    initialize_db,
)

logger = logging.getLogger(__name__)


def source_label(doc) -> str:
    """Build a human-readable source location for a retrieved document."""
    meta = doc.metadata
    parts = [Path(meta.get("source", "unknown")).name]

    if "page" in meta:
        parts.append(f"page {meta['page']}")
    if "start_line" in meta:
        parts.append(f"lines {meta['start_line']}-{meta['end_line']}")

    return ", ".join(parts)


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Loads and validates all parameters from config.yaml.

    Reads ``config.yaml`` (the single source of truth) and validates it
    against :class:`raggy.config.RaggySettings`, which enforces types, ranges,
    allowed enum values, and cross-field invariants (e.g. ``retrieve_k`` vs
    ``rerank_k``). Returns a plain dict for use by the rest of the library.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Runtime config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    settings = RaggySettings(**cfg)

    result = settings.model_dump()
    # Preserve the historical public behavior: rerank_k defaults to retrieve_k.
    if result["rerank_k"] is None:
        result["rerank_k"] = result["retrieve_k"]
    return result


def _init_db() -> Chroma:
    cfg = load_config()

    # Embeddings are always local, so the (always-required) embedding model is
    # pulled up front before any embedding work begins.
    ensure_ollama_model(cfg["embedding_model"])

    vectorstore = initialize_db(
        persist_directory=cfg["persist_directory"],
        embedding_model=cfg["embedding_model"],
        sources=cfg["sources"],
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        batch_size=cfg["batch_size"],
    )

    return vectorstore


_vectorstore: Chroma | None = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _init_db()
    return _vectorstore


def refresh_db(cfg: dict[str, Any] | None = None) -> bool:
    """Re-check the source docs and re-index the vector DB if they changed.

    ``_get_vectorstore`` caches the initialized store for the life of the
    process, so interactive sessions (like the CLI) would otherwise silently
    keep using a stale DB. Call this before each query; it returns True if the
    DB was re-indexed (incrementally when only the source files changed, see
    :func:`raggy.vectorstore.initialize_db`).

    Pass a pre-loaded ``cfg`` dict to avoid re-reading ``config.yaml``
    (and re-hashing source files) when the caller already has it.
    """
    global _vectorstore
    if cfg is None:
        cfg = load_config()
    index_cfg = build_index_config(
        sources=cfg["sources"],
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        embedding_model=cfg["embedding_model"],
    )
    if not db_needs_rebuild(cfg["persist_directory"], index_cfg):
        return False

    logger.info("Source documents or index config changed; re-indexing DB.")
    if _vectorstore is not None:
        close_vectorstore(_vectorstore)
    _vectorstore = _init_db()
    return True


def run_pipeline(query: str, chat_history: list | None = None) -> tuple[Any, list[Any]]:
    """Run ``query`` through the RAG pipeline and return (answer, retrieved docs).

    ``chat_history`` optionally carries prior turns as ``("human"|"ai", content)``
    tuples so the answer can build on the conversation; the chain is then
    history-aware (the question is condensed for retrieval, and the history is
    part of the prompt).

    This is a library function: errors (missing config, missing Ollama model,
    missing provider API key, embedding failures, ...) propagate to the caller
    instead of terminating the process. CLI wrappers are responsible for
    catching and reporting them.
    """
    cfg = load_config()
    vectorstore = _get_vectorstore()

    # Build the RAG chain and retrieve retriever. When generation is local,
    # ensure the configured LLM is present in Ollama before constructing it.
    if cfg["llm_provider"] == "ollama":
        ensure_ollama_model(cfg["llm_model"])

    doc_sink: list = []
    rag_chain, _ = build_rag_chain(
        vectorstore=vectorstore,
        llm_model=cfg["llm_model"],
        llm_provider=cfg["llm_provider"],
        system_prompt=cfg["system_prompt"],
        retrieve_k=cfg["retrieve_k"],
        temperature=cfg["temperature"],
        rerank_enabled=cfg["rerank_enabled"],
        rerank_model=cfg["rerank_model"],
        rerank_k=cfg["rerank_k"],
        rerank_threshold=cfg["rerank_threshold"],
        persist_directory=cfg["persist_directory"],
        hybrid_search=cfg["hybrid_search"],
        hybrid_alpha=cfg["hybrid_alpha"],
        doc_sink=doc_sink,
        chat_history=chat_history,
    )

    # Run query through RAG pipeline. The retrieved sources are captured
    # inside the chain in a single pass (doc_sink), so they match exactly
    # the context the LLM used to answer.
    response = rag_chain.invoke({"question": query, "chat_history": chat_history or []})

    # Output the source retrieved documents for verification. Full
    # Document objects are returned so callers can inspect the metadata
    # (source file, page, line range) attached to each retrieved chunk.
    retrieved_docs = doc_sink[-1] if doc_sink else []

    return response, retrieved_docs


def run_pipeline_stream(
    query: str,
    doc_sink: list | None = None,
    chat_history: list | None = None,
) -> Iterator[str]:
    """Run ``query`` through the RAG pipeline, yielding answer text chunks.

    Mirrors ``run_pipeline`` but streams the LLM output token-by-token so
    callers (like the CLI) can render it incrementally. The exact context the
    LLM saw is appended to ``doc_sink`` in a single pass (see
    ``build_rag_chain``); pass a list in and read ``doc_sink[-1]`` after
    consuming the stream. ``chat_history`` has the same meaning as in
    ``run_pipeline``. Errors propagate to the caller.
    """
    cfg = load_config()
    vectorstore = _get_vectorstore()

    if cfg["llm_provider"] == "ollama":
        ensure_ollama_model(cfg["llm_model"])

    collected: list = [] if doc_sink is None else doc_sink
    rag_chain, _ = build_rag_chain(
        vectorstore=vectorstore,
        llm_model=cfg["llm_model"],
        llm_provider=cfg["llm_provider"],
        system_prompt=cfg["system_prompt"],
        retrieve_k=cfg["retrieve_k"],
        temperature=cfg["temperature"],
        rerank_enabled=cfg["rerank_enabled"],
        rerank_model=cfg["rerank_model"],
        rerank_k=cfg["rerank_k"],
        rerank_threshold=cfg["rerank_threshold"],
        persist_directory=cfg["persist_directory"],
        hybrid_search=cfg["hybrid_search"],
        hybrid_alpha=cfg["hybrid_alpha"],
        doc_sink=collected,
        chat_history=chat_history,
    )

    yield from rag_chain.stream({"question": query, "chat_history": chat_history or []})
