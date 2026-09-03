# raggy

A lightweight Retrieval-Augmented Generation (RAG) package built with
LangChain and Chroma. It is local-first: your documents and the vector DB live
on your machine, embeddings run locally via Ollama, and answer
generation can run either locally (Ollama) or via a cloud provider's API. It supports
most common formats and uses OCR automatically when needed.

These are all supported file formats (all other formats are ignored):

`.pdf`, `.docx`, `.pptx`, `.txt`, `.md`, `.markdown`, `.html`, `.htm`, `.png`, `.jpg`, `.jpeg`, `.bmp`.


## Prerequisites

Ollama is required for running the local embedding model (which feeds the on-disk vector DB), and also local LLM (if needed). To install Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

You also may need to start Ollama explicitly:

```bash
ollama serve
```

If API key will be used for accessing LLM remotely, a standard environment variable needs to be set (which is one of):

```bash
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

You also need Python version 3.10 or higher, having `uv` installed is recommended:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installing `raggy`

Clone the repo:

```bash
git clone https://github.com/paulknysh/raggy.git && cd raggy
```

To install CLI only:

```bash
# with uv
uv tool install -e .
# with pip
pip install -e .
```

To use raggy programmatically (as a library), or to modify/test code:

```bash
uv sync
source .venv/bin/activate

# to run lint/format/tests
make sure
```

## Usage

First, run this command:

```bash
make config
```

It creates your own user config (`config.yaml`). This is where all your document paths, LLM proviers, models, and other execution parameters live. For detailed overview of all config parameters see [Configuration](#configuration). Your `sources` section will look like this:

```
sources:
  - "/Users/xyz/Documents/docs" # folder
  - "/Users/xyz/Downloads/docs_1" # folder
  - "/Users/xyz/Desktop/docs_2/abc.pdf" # file
```

To start CLI type (by default always uses `config.yaml` created inside repo):

```bash
raggy
```

To run with a config file located elsewhere, pass `--config`:

```bash
raggy --config path/to/config.yaml
```

**Important:** On the first run CLI automatically pulls Ollama models listed in `config.yaml` and indexes your
documents -- this might take a while, depending on models chosen, document count/size and whether OCR is needed (scans, images etc)

<img src="assets/cli_demo.png">

You can also use it programmatically:

```python
from raggy import run_pipeline, source_label

query = "What is TS-RAG?"

response, retrieved_docs = run_pipeline(query)

print(f"\n*** RESPONSE:\n\n{response}\n\n***")

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n\n=== Doc {i} [{source_label(doc)}] ===\n\n")
    print(doc.page_content)
```

## Configuration

All runtime settings are read from `config.yaml` (the file must be present in the
working directory when you run the `raggy` CLI or import the library, unless you
point at another one with `raggy --config PATH` or the `config_path` argument of
`run_pipeline`):

| Setting | Description |
| --- | --- |
| `sources` | list of source directories and/or files |
| `persist_directory` | location where the DB itself is stored |
| `embedding_model` | Ollama embedding model (e.g. `nomic-embed-text`) |
| `chunk_size` | chunk size in characters |
| `chunk_overlap` | character overlap between adjacent chunks |
| `batch_size` | max number of chunks embedded per batch into Chroma (default `100`); the number of batches is derived dynamically from the chunk count |
| `llm_provider` | where generation runs: `ollama` (local, default) or `openai`/`anthropic`/`google` (via API) |
| `llm_model` | chat model for generation (e.g. `phi4-mini` locally, or a remote model name like `gemini-3.7-flash`) |
| `temperature` | LLM sampling temperature |
| `retrieve_k` | number of chunks returned by the first-stage retrieval |
| `hybrid_search` | whether to fuse dense retrieval with a lexical BM25 (`bm25s`) pass via reciprocal rank fusion (on by default) |
| `hybrid_alpha` | weight of the dense/vector pass in hybrid fusion (`1.0` = vector only, `0.0` = BM25 only, default `0.5`) |
| `rerank_enabled` | whether to use additional reranking after retrieval stage (on by default) |
| `rerank_model` | Hugging Face ID of the cross-encoder model |
| `rerank_k` | final number of chunks returned by the cross-encoder (must be `<= retrieve_k`) |
| `rerank_threshold` | drop reranked chunks whose relevance score is below this value (`0.0` = disabled, `0.3` by default) |
| `system_prompt` | system prompt dictating how the LLM should answer; must contain a `{context}` placeholder |

## Demo dataset

This article (https://arxiv.org/abs/2608.06223v1) is used here for testing. It's an 8-page document -- each page
is saved in different file formats (including PDF, plaintext, images, MS Word) and saved inside
`sample_docs` directory. This directory is specified in `config.yaml` by default.

## Demo eval

`eval` folder currently contains simple Q&A dataset to test pipeline performance on demo dataset end-to-end.
To run the test, start the Ollama service with the embeddings always local and models pulled automatically, then run:

```bash
uv run eval/run_eval.py
```

It computes basic retrieval/generation metrics and produces a summary (both printed and saved to `eval/results.json`). The Q&A pairs are about `sample_docs`, so the harness always runs against `default_config/default_config.yaml` rather than your own `config.yaml`.


## The DB management

The DB lives in the directory configured by `persist_directory`
(default `./db`), which holds both Chroma vector store and BM25 index.

### How the DB is created

On the first run, `initialize_db` (in `raggy/indexing.py`) loads every
supported file under each entry in `sources`, splits them into overlapping chunks,
and embeds them into Chroma. It also writes a `manifest.yaml` into the persist
directory recording these parameters:
- `sources`
- `chunk_size`
- `chunk_overlap`
- `embedding_model`
- `files` — a `{file path: SHA-256 content hash}` map of every indexed file

The same chunks are also indexed with `bm25s` (a lexical BM25 index), stored in
`<persist_directory>/bm25_index/`, so `hybrid_search` can run without re-indexing
at retrieval time.

### How/when the DB is re-indexed

The `manifest.yaml` is cheap to recompute, so for every new run it is computed and
compared against the existing one. What happens next depends on what changed:

- **Incremental update (the common case)** — the source files changed but
  `chunk_size`, `chunk_overlap`, and `embedding_model` did not. Comparing the two
  `files` maps names exactly which files were added, modified, or deleted. Chunks
  belonging to deleted and modified files are dropped from Chroma, and only added
  and modified files are re-loaded and embedded. Untouched files are never
  re-embedded, so editing one file in a large corpus costs one file's worth of work.
- **Full rebuild** — `chunk_size`, `chunk_overlap`, or `embedding_model` changed
  (every stored vector is then invalid), or there is no manifest yet. The persist
  directory is wiped and everything is indexed from scratch. Note that a manifest
  written before incremental indexing existed also triggers one full rebuild.

The BM25 index has no incremental update path, so it is rebuilt after every update —
from the chunks already stored in Chroma, which needs no embedding calls and no
re-reading of source files.


## Notes

- `PyPDFLoader` parses PDFs page-by-page (each Document carries a `page` key);
  PowerPoint decks are parsed slide-by-slide (also exposed as `page`); plain-text
  files (`txt`/`md`/`markdown`) get approximate `start_line`/`end_line`
  annotations via `annotate_line_numbers`. HTML, Word documents (`.docx`) and
  images carry no location metadata — HTML because `BSHTMLLoader` returns the
  extracted text rather than the file, so lines counted in it do not match the
  source; DOCX because it has no native page boundaries (Word's pagination can't
  be reproduced reliably); images because they have no meaningful lines.
- Image files (`.png`/`.jpg`/`.jpeg`/`.bmp`) are OCR'd with RapidOCR (runs fully
  on-device via ONNX Runtime); PDFs with no extractable text layer are detected
  and automatically OCR'd page-by-page as well.
- The `langchain-community` loader deprecation warning is suppressed in the
  pytest config.

## TODOs

- [x] Support of popular LLM providers via API keys
- [x] Conversation memory in chat mode
- [x] Pydantic validation of config file
- [x] Hybrid retrieval, tuning config defaults
- [x] Incremental indexing (only re-embed files that changed)
- [ ] UX/UI tuning of CLI (improved commands/statuses etc)
- [ ] Performance optimizations (DB creation, pipeline itself)

## License

MIT
