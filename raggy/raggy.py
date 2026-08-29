import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from langchain_chroma import Chroma

from .pipeline import build_rag_chain
from .reranker import DEFAULT_RERANKER_MODEL
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
    """Loads all parameters from config.yaml (the single source of truth)."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Runtime config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    sources = cfg["sources"]
    if isinstance(sources, str):
        sources = [sources]
    if not sources:
        raise ValueError("'sources' must contain at least one file or directory")

    retrieve_k = int(cfg["retrieve_k"])
    rerank_k = int(cfg.get("rerank_k", retrieve_k))
    if rerank_k > retrieve_k:
        raise ValueError(
            f"rerank_k ({rerank_k}) must not be greater than retrieve_k ({retrieve_k})"
        )

    return {
        "sources": [str(source) for source in sources],
        "persist_directory": str(cfg["persist_directory"]),
        "chunk_size": int(cfg["chunk_size"]),
        "chunk_overlap": int(cfg["chunk_overlap"]),
        "batch_size": int(cfg["batch_size"]),
        "embedding_model": str(cfg["embedding_model"]),
        "llm_provider": str(cfg.get("llm_provider", "ollama")),
        "llm_model": str(cfg["llm_model"]),
        "temperature": float(cfg["temperature"]),
        "retrieve_k": retrieve_k,
        "mmr_fetch_k": int(cfg["mmr_fetch_k"]),
        "search_type": str(cfg["search_type"]),
        "rerank_enabled": bool(cfg.get("rerank_enabled", False)),
        "rerank_model": str(cfg.get("rerank_model", DEFAULT_RERANKER_MODEL)),
        "rerank_k": rerank_k,
        "system_prompt": str(cfg["system_prompt"]),
    }


def _init_db() -> Chroma:
    cfg = load_config()

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


def refresh_db() -> bool:
    """Re-check the source docs and rebuild the vector DB if they changed.

    ``_get_vectorstore`` caches the initialized store for the life of the
    process, so interactive sessions (like the CLI) would otherwise silently
    keep using a stale DB. Call this before each query; it returns True if the
    DB was rebuilt.
    """
    global _vectorstore
    cfg = load_config()
    index_cfg = build_index_config(
        sources=cfg["sources"],
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        embedding_model=cfg["embedding_model"],
    )
    if not db_needs_rebuild(cfg["persist_directory"], index_cfg):
        return False

    logger.info("Source documents or index config changed; rebuilding DB.")
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

    # Build the RAG chain and retrieve retriever
    doc_sink: list = []
    rag_chain, _ = build_rag_chain(
        vectorstore=vectorstore,
        llm_model=cfg["llm_model"],
        llm_provider=cfg["llm_provider"],
        system_prompt=cfg["system_prompt"],
        search_type=cfg["search_type"],
        k=cfg["retrieve_k"],
        fetch_k=cfg["mmr_fetch_k"],
        temperature=cfg["temperature"],
        rerank_enabled=cfg["rerank_enabled"],
        rerank_model=cfg["rerank_model"],
        rerank_k=cfg["rerank_k"],
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

    collected: list = [] if doc_sink is None else doc_sink
    rag_chain, _ = build_rag_chain(
        vectorstore=vectorstore,
        llm_model=cfg["llm_model"],
        llm_provider=cfg["llm_provider"],
        system_prompt=cfg["system_prompt"],
        search_type=cfg["search_type"],
        k=cfg["retrieve_k"],
        fetch_k=cfg["mmr_fetch_k"],
        temperature=cfg["temperature"],
        rerank_enabled=cfg["rerank_enabled"],
        rerank_model=cfg["rerank_model"],
        rerank_k=cfg["rerank_k"],
        doc_sink=collected,
        chat_history=chat_history,
    )

    yield from rag_chain.stream({"question": query, "chat_history": chat_history or []})
