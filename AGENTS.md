# AGENTS.md

## Project
`raggy` — a local-first RAG package (Python 3.10+, LangChain + Chroma + Ollama). Embeddings and the vector DB are always local; generation can be local (Ollama) or remote (OpenAI/Anthropic/Google API).

## Commands (run via `uv`)
- Lint + Format + Test: `make sure`

## Runtime prerequisites (not mocked in tests)
- `config.yaml` must exist in the working directory (single source of truth, validated in `raggy/raggy.py:load_config`).
- Ollama service running with the embedding model named in `config.yaml` (`nomic-embed-text`) — always required because embeddings are local. The generation LLM is local (`llama3.2`) when `llm_provider: ollama`, or a remote API when `llm_provider: openai|anthropic|google`. Tests mock all Ollama/Chroma/provider I/O — never require a live service in tests.

## Architecture
- `raggy/raggy.py` — orchestration: `load_config`, `run_pipeline`, `run_pipeline_stream`, `refresh_db`; caches the vectorstore in a module global.
- `raggy/pipeline.py` — LCEL chain: retriever → LLM relevance filter → `format_and_capture` (appends docs to `doc_sink`) → prompt → LLM.
- `raggy/vectorstore.py` — Chroma init/ingest/batching + manifest-based rebuild detection (ANY change to sources/chunk_size/chunk_overlap/embedding_model/content_hash ⇒ full wipe+reindex).
- `raggy/loaders.py` — format loaders; OCR (RapidOCR) for images and textless PDFs; unsupported files ignored not errored.
- `raggy/reranker.py` — onnxruntime cross-encoder, cached globally (no PyTorch).
- `raggy/llm_factory.py` — provider-agnostic `get_llm`; dispatches on `llm_provider` to Ollama (local) or OpenAI/Anthropic/Google (API, key from env var).
- `raggy/llm_filter.py` — LLM-based relevance filter (`filter_docs_by_relevance`); single chat-LLM call flags each retrieved chunk yes/no, keeps only "yes" in original order; fail-open and strict.
- `raggy/cli.py` — rich-based chat CLI; catches errors that library functions deliberately propagate.

## Conventions
- Add a docstring to each public function; keep released helper logic in the module where it's used.
- No code comments unless the code warrants an explanation (a few exist for subtle bugs, e.g. `close_vectorstore` before DB wipe).
- Public API exported from `raggy/__init__.py` is `load_config`, `run_pipeline`, `source_label`.
- Tests live in `tests/`, one file per module; UX/eval harness in `eval/`.
