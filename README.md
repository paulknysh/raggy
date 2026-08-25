# raggy

A lightweight Retrieval-Augmented Generation (RAG) package built with LangChain, Chroma, and Ollama.

Runs fully locally, supports common document formats (PDF, MS Word/Powerpoint, plain text, images) and uses OCR automatically when needed.

Mainly built for self-education and experimenting.


## Setting up Ollama

Ollama runs local models with no API key required. Install it for your platform,
then pull the models referenced in `config.yaml`.

Install Ollama (macOS / Linux):

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Starting the Ollama service:

```bash
ollama serve
```

Then pull the models configured in `config.yaml`:

```bash
ollama pull nomic-embed-text   # embedding model (embeddings)
ollama pull llama3.2           # chat LLM (generation)
```

`nomic-embed-text` is a lightweight embedding model used to index document chunks;
`llama3.2` is the chat model that generates answers from the retrieved context.
Once downloaded they are cached locally, so future runs use them immediately.

## Installing `raggy`

Requires Python 3.10 or newer.

### Install with uv (recommended)

`uv` creates and manages the virtual environment for you. For a normal
install use `uv sync`; for development (adds `pytest`/`ruff`) use the `dev`
extra — this is also what CI runs:

```bash
uv sync --extra dev
```

### Install with pip / uv pip

These commands require an **active virtual environment** (`uv pip` will not
create one). Inside your venv:

```bash
pip install .
# or
uv pip install .
```

For an editable/development install (adds the `dev` extra with `pytest`/`ruff`):

```bash
pip install -e ".[dev]"
# or
uv pip install -e ".[dev]"
```

## Configuration

All runtime settings are read from `config.yaml` (the file must be present in the
working directory when you run the `raggy` CLI or import the library):

- `sources` — list of directories and/or files to index. The following file formats are supported,
 (all others are ignored): `.txt`, `.md`, `.markdown`, `.pdf`, `.docx`, `.pptx`, `.html`, `.htm`, `.png`, `.jpg`, `.jpeg`, `.bmp`.
- `persist_directory` — location where the DB itself is stored
- `embedding_model` — Ollama embedding model (e.g. `nomic-embed-text`)
- `chunk_size` — chunk size in characters
- `chunk_overlap` — character overlap between adjacent chunks
- `n_batches` — number of batches used when embedding chunks into Chroma
- `llm_model` — Ollama chat model (e.g. `llama3.2`)
- `temperature` — LLM sampling temperature
- `search_type` — retriever search type (e.g. `mmr` or `similarity`)
- `retrieve_k` — number of chunks returned by the first-stage retrieval
- `mmr_fetch_k` — number of candidate chunks fetched before MMR reranking (used when `search_type: mmr`)
- `rerank_enabled` — whether to use reranking after retrieval stage
- `rerank_model` — Hugging Face ID of the cross-encoder model (default `cross-encoder/ms-marco-MiniLM-L6-v2`)
- `rerank_k` — final number of chunks returned by the cross-encoder (must be `<= retrieve_k`)
- `system_prompt` — system prompt dictating how the LLM should answer; must contain a `{context}` placeholder


## Usage

Use `raggy` command to run in interactive CLI mode.

Or use it as a library:

```python
from raggy import run_pipeline, source_label

query = "What is TS-RAG?"

response, retrieved_docs = run_pipeline(query)

print(f"\n*** RESPONSE:\n\n{response}\n\n***")

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n\n=== Doc {i} [{source_label(doc)}] ===\n\n")
    print(doc.page_content)
```


## Default dataset

This article (https://arxiv.org/abs/2608.06223v1) is used here for testing. It's an 8-page document -- each page
is saved in different file formats (including PDF, plaintext, images, MS Word) and saved inside
`sample_docs_pt1` and `sample_docs_pt2` directories. These directories are specified in `config.yaml` by default.

## Default eval

`eval` folder currently contains simple Q&A dataset to test pipeline performance end-to-end.
To run the test, make sure the Ollama service is running and the configured models are pulled, then run:

```bash
uv run eval/run_eval.py
```

It computes basic retrieval/generation metrics and produces a summary (both printed and saved to `eval/results.json`).


## The DB management

The Chroma DB lives in the directory configured by `persist_directory`
(default `./chroma_db`). It is (re-)indexed automatically on each run as needed.

### How the DB is created

On the first run, `initialize_db` (in `raggy/vectorstore.py`) loads every
supported file under each entry in `sources`, splits them into overlapping chunks,
and embeds them into Chroma. It also writes a `manifest.yaml` into the persist
directory recording these parameters:
- `sources`
- `chunk_size`
- `chunk_overlap`
- `embedding_model`
- `content_hash` — a SHA-256 fingerprint of the source files

### How/when the DB is re-indexed

The `manifest.yaml` is cheap to recompute, so for every new run it is computed and compared against existing one.
Any changes to its content, including changes to `content_hash` will cause DB to be re-indexed.


## Testing, linting, and formatting

After installing the dev extras (see above, e.g. `uv sync --extra dev`):

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format .
```

CI (`.github/workflows`) runs lint, format check, and tests on every push.

## License

MIT (see `LICENSE`).

## Notes

- `PyPDFLoader` parses PDFs page-by-page (each Document carries a `page` key);
  PowerPoint decks are parsed slide-by-slide (also exposed as `page`); text-based
  files (`txt`/`md`/`markdown`/`html`) get approximate `start_line`/`end_line`
  annotations via `annotate_line_numbers`. Word documents (`.docx`) and images
  carry no location metadata — DOCX has no native page boundaries (Word's
  pagination can't be reproduced reliably) and images have no meaningful lines.
- Image files (`.png`/`.jpg`/`.jpeg`/`.bmp`) are OCR'd with RapidOCR (runs fully
  on-device via ONNX Runtime); PDFs with no extractable text layer are detected
  and automatically OCR'd page-by-page as well.
- The `langchain-community` loader deprecation warning is suppressed in the
  pytest config.

## Planned TODOs

- More optimizations (improved runtime, incremental DB re-indexing etc)
- `config.yaml` validation
- Conversation memory in chat mode
- Hybrid retrieval, additional pipeline tuning
