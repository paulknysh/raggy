# AGENTS.md

## Project
`raggy` — a local-first RAG package (Python 3.10+, LangChain + Chroma + Ollama). Embeddings and the vector DB are always local; generation can be local (Ollama) or remote (OpenAI/Anthropic/Google API).

## Commands (run via `uv`)
- Lint + Format + Test: `make sure`

## Runtime prerequisites (not mocked in tests)
- `config.yaml` must exist in the working directory (single source of truth, validated in `raggy/raggy.py:load_config`).
- Ollama service running with the embedding model named in `config.yaml` (`nomic-embed-text`) — always required because embeddings are local. The generation LLM is local (`phi4-mini`) when `llm_provider: ollama`, or a remote API when `llm_provider: openai|anthropic|google`. Tests mock all Ollama/Chroma/provider I/O — never require a live service in tests.

## Architecture
- `raggy/raggy.py` — orchestration: `load_config`, `run_pipeline`, `run_pipeline_stream`, `refresh_db`; caches the vectorstore in a module global.
- `raggy/pipeline.py` — LCEL chain: hybrid retriever (dense + optional BM25 when `hybrid_search: true`, fused via `EnsembleRetriever`) → (optional) cross-encoder reranker → score-threshold filter (drops chunks below `rerank_threshold`, no-op when reranking is off) → `format_and_capture` (appends docs to `doc_sink`) → prompt → LLM.
- `raggy/vectorstore.py` — Chroma init/ingest/batching + manifest-based staleness detection. The manifest holds a per-file `{path: content hash}` map, so `plan_index_update` returns an `IndexPlan` naming added/modified/deleted files: a change to chunk_size/chunk_overlap/embedding_model (or a missing/legacy manifest) ⇒ full wipe+reindex, source-file-only changes ⇒ incremental `update_index` (delete stale chunks by `source` metadata, embed only added/modified files). Also builds a `bm25s` index for hybrid retrieval, persisted to `<persist_directory>/bm25_index`; it has no incremental path and is rebuilt from the chunks stored in Chroma after each update.
- `raggy/bm25_retriever.py` — `Bm25sRetriever` (a LangChain `BaseRetriever`); loads the persisted `bm25s` index + per-chunk metadata and returns `Document`s for the lexical pass of hybrid retrieval.
- `raggy/loaders.py` — format loaders; `load_documents_from_sources` (walk the configured sources) and `load_documents_from_paths` (an explicit file list, used by incremental indexing); OCR (RapidOCR) for images and textless PDFs; unsupported files ignored not errored.
- `raggy/reranker.py` — onnxruntime cross-encoder, cached globally (no PyTorch).
- `raggy/score_filter.py` — `ScoreAnnotatingReranker` stamps each reranked chunk with its cross-encoder score (`relevance_score`); `filter_by_score_threshold` drops chunks below the configurable `rerank_threshold` before the final generation step.
- `raggy/llm_factory.py` — provider-agnostic `get_llm`; dispatches on `llm_provider` to Ollama (local) or OpenAI/Anthropic/Google (API, key from env var).
- `raggy/cli.py` — rich-based chat CLI; catches errors that library functions deliberately propagate.

## Conventions
- Add a docstring to each public function; keep released helper logic in the module where it's used.
- No code comments unless the code warrants an explanation (a few exist for subtle bugs, e.g. `close_vectorstore` before DB wipe).
- Public API exported from `raggy/__init__.py` is `load_config`, `run_pipeline`, `source_label`.
- Tests live in `tests/`, one file per module; UX/eval harness in `eval/`.
