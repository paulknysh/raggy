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
- `raggy/vectorstore.py` — Chroma init/ingest/batching + manifest-based staleness detection. The manifest holds a per-file `{path: content hash}` map, so `plan_index_update` returns an `IndexPlan` naming added/modified/deleted files: a change to chunk_size/chunk_overlap/embedding_model (or a missing/legacy manifest) ⇒ full wipe+reindex, source-file-only changes ⇒ incremental `update_index` (delete stale chunks by `source` metadata, embed only added/modified files). Also refreshes the BM25 index for hybrid retrieval via `save_bm25_index` (see `bm25_retriever.py`); it has no incremental path and is rebuilt from the chunks stored in Chroma after each update.
- `raggy/bm25_retriever.py` — the whole BM25 side of hybrid retrieval: `save_bm25_index` writes the `bm25s` index + per-chunk metadata to `<persist_directory>/bm25_index` at build time, and `Bm25sRetriever` (a LangChain `BaseRetriever`) loads them back and returns `Document`s for the lexical pass.
- `raggy/loaders.py` — format loaders; `source_files` is the corpus's **single walk** (recursive, ordered, deduplicated across overlapping entries) and both `load_documents` and `vectorstore.file_fingerprints` consume it, so the manifest's keys are by construction the `source` metadata written on the chunks — never re-implement the walk elsewhere. `load_documents(sources, progress, on_missing)` is the one loader: `on_missing="raise"` (default) for the configured sources, where a missing path is a config error; `on_missing="skip"` for incremental indexing, whose file list was fingerprinted earlier in the same run so a since-deleted file must not abort the update. OCR (RapidOCR) for images and textless PDFs. Unsupported files and source directories holding none are ignored, not errored; an unreadable file is logged and skipped.
- `raggy/progress.py` — the `ProgressCallback` Protocol shared by the slow steps (model pulls, ingest, embedding): `(message, completed=None, total=None)`, where the counters are only passed for byte-measured work.
- `raggy/reranker.py` — onnxruntime cross-encoder, cached globally (no PyTorch). `ensure_reranker_model` prefetches the ONNX weights + tokenizer into the huggingface-hub cache; with a `progress` callback it hands `hf_hub_download` a tqdm stand-in (`_progress_tqdm_class`) so the bytes land on the caller's status line instead of hf/Xet drawing its own bars.
- `raggy/score_filter.py` — `ScoreAnnotatingReranker` stamps each reranked chunk with its cross-encoder score (`relevance_score`); `filter_by_score_threshold` drops chunks below the configurable `rerank_threshold` before the final generation step.
- `raggy/llm_factory.py` — provider-agnostic `get_llm`; dispatches on `llm_provider` to Ollama (local) or OpenAI/Anthropic/Google (API, key from env var). `ensure_ollama_model` checks `ollama list` and pulls what is missing: via `ollama pull` (Ollama draws its own bars) when no `progress` callback is given, otherwise over the Ollama API (`_stream_pull`) so the caller renders the download. `raggy.py:ensure_models` pulls everything a config needs locally (embedding model always, LLM only when `llm_provider: ollama`, reranker cross-encoder when `rerank_enabled`).
- `raggy/cli.py` — rich-based chat CLI; catches errors that library functions deliberately propagate. `_ProgressLine` keeps every slow step on one in-place line: a spinner + message for text-only work (`[i/N] ingesting <file> ...`, embedding batches) and a download bar for model pulls; the message strings come from the library through the optional `progress` callback (`raggy/progress.py`), threaded from `ensure_models`/`refresh_db` down to `llm_factory` and the loaders.

## Conventions
- Add a docstring to each public function; keep released helper logic in the module where it's used.
- No code comments unless the code warrants an explanation (a few exist for subtle bugs, e.g. `close_vectorstore` before DB wipe).
- Public API exported from `raggy/__init__.py` is `load_config`, `run_pipeline`, `source_label`.
- Tests live in `tests/`, one file per module; UX/eval harness in `eval/`.
