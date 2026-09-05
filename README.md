# raggy

A lightweight Retrieval-Augmented Generation (RAG) CLI tool built with LangChain, Chroma, and Ollama. Hybrid database (vector + BM25 index) and embedding generation run fully locally. Answer generation can run either via a local LLM or using a cloud provider's API. The tool supports most common document formats and uses OCR automatically when needed.

These are all currently supported file formats (all other formats are ignored):

`.pdf`, `.docx`, `.pptx`, `.txt`, `.md`, `.markdown`, `.html`, `.htm`, `.png`, `.jpg`, `.jpeg`, `.bmp`.


## Prerequisites

Ollama is required for running the local embedding model (which feeds the on-disk vector DB), and also a local LLM (if needed). To install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh

# may need to start ollama after installation using the app or:
ollama
# or
ollama serve
```

If an API key will be used for accessing an LLM remotely, a standard environment variable needs to be set (one of):

```bash
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

Python 3.10 or newer is required; installing `uv` is recommended:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

Clone the repo:

```bash
git clone https://github.com/paulknysh/raggy.git && cd raggy
```

To install the CLI tool only:

```bash
# with uv
uv tool install -e .

# with pipx
pipx install -e .
```

To be able to use raggy programmatically (as a library):

```bash
uv sync

# if you modify/test code, this runs lint/format/tests
make sure
```

## Usage (CLI)

First, run this command:

```bash
make config
```

It creates your own user config (`config.yaml`) where all your execution parameters live. For a detailed overview of all config parameters, see [Configuration](#configuration).

`config.yaml` comes with defaults you can test. Populate `sources` (your input folders/files) and `db_directory` (DB location) sections with your preferred paths.

**Important:** Relative paths in `config.yaml` resolve against the current working directory, so keep that in mind if you want to run CLI tool from other locations. To be safe, just always use absolute paths in config.

To start the CLI, use the `raggy` command followed by the path to your config file:

```bash
raggy config.yaml
```

The path can point anywhere (e.g. `raggy path/to/other_config.yaml`), so several configs can live side by side:

**Important:** CLI automatically pulls all models listed in `config.yaml` and (re-)indexes your documents -- this might take a while on the first run, depending on models chosen, document count/size and whether OCR is needed (scans, images etc).

Below is a screenshot of the CLI. For each query, it returns an answer, citations, their corresponding relevance scores, and locations:

<img src="assets/cli_demo.png">

## Usage (programmatic)

Here is the basic snippet you can run via `uv run snippet.py`:

```python
from raggy import run_pipeline, source_label

query = "What is TS-RAG?"

response, retrieved_docs = run_pipeline(query, config_path="config.yaml")

print(f"\n*** RESPONSE:\n\n{response}\n\n***")

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n\n=== Doc {i} [{source_label(doc)}] ===\n\n")
    print(doc.page_content)
```

## Configuration

All runtime settings available in config file:

| Setting | Description |
| --- | --- |
| `sources` | list of source directories and/or files |
| `db_directory` | location where the DB itself is stored |
| `embedding_model` | Ollama embedding model (e.g. `nomic-embed-text`) |
| `chunk_size` | chunk size in characters |
| `chunk_overlap` | character overlap between adjacent chunks |
| `embed_batch_size` | max number of chunks embedded per batch into Chroma (`100` in the shipped config); the number of batches is derived dynamically from the chunk count |
| `llm_provider` | where generation runs: `ollama` (local, the shipped value) or `openai`/`anthropic`/`google` (via API) |
| `llm_model` | chat model for generation (e.g. `phi4-mini` locally, or a remote model name like `gemini-3.7-flash`) |
| `llm_temperature` | LLM sampling temperature |
| `retrieve_k` | total chunks retrieved per query, split across the dense and BM25 passes (`50` in the shipped config) |
| `hybrid_alpha` | share of `retrieve_k` spent on the dense/vector pass; the remainder goes to BM25 (`1.0` = vector only, `0.0` = BM25 only, `0.5` in the shipped config) |
| `rerank_model` | Hugging Face ID of the cross-encoder model |
| `rerank_k` | final number of chunks returned by the cross-encoder (must be `<= retrieve_k`) |
| `rerank_threshold` | drop reranked chunks whose relevance score is below this value (`0.0` = disabled, `0.3` in the shipped config) |
| `system_prompt` | system prompt dictating how the LLM should answer; must contain a `{context}` placeholder |

Every setting above is required, and unknown keys are rejected. There are no
fallback values in the code: `config.yaml` is the complete runtime state, and
the values quoted above are simply what `make config` writes -- not what you
get by leaving a setting out. Omitting a setting, misspelling one, or leaving
behind a key from an older version all fail at startup with the offending
names listed, rather than silently running with a value you did not choose.

Retrieval always runs in two stages, so there is nothing to switch on. Every
query is answered by a hybrid search -- a dense pass over the vector store plus
a lexical BM25 pass -- whose results are merged by reciprocal rank fusion and
then re-scored by a cross-encoder.

The two numbers worth knowing are `retrieve_k` and `rerank_k`. `retrieve_k` is
the total candidate budget and the main cost lever, since the cross-encoder
scores every candidate it returns; `rerank_k` is how many chunks survive into
the prompt, and is the number to raise if answers look like they are missing
context. `hybrid_alpha` only decides how the fixed `retrieve_k` budget is
divided between the two passes, so moving it costs nothing: lower it for
corpora where exact tokens matter (code, identifiers, part numbers, legal
citations) and raise it for prose where questions paraphrase the source.

## Demo dataset

This article (https://arxiv.org/abs/2608.06223v1) is used here for testing. It's an 8-page document -- each page is saved in different file formats (including PDF, plaintext, images, MS Word) and saved inside the `sample_docs` directory. This directory is specified in `config.yaml` by default.

## Demo eval

`eval` folder currently contains a basic harness to test pipeline performance on the demo dataset. You can run it by:

```bash
uv run eval/run_eval.py
```

It computes basic retrieval/generation metrics and produces a summary (both printed and saved to `eval/results.json`). The Q&A pairs are about `sample_docs`, so the harness always runs against `default_config/default_config.yaml` rather than your own `config.yaml`.


## The DB management

DB creates/updates itself automatically, so you don't need to think about it. Below is just a high-level overview of the mechanics.

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
`<db_directory>/bm25_index/`, so the lexical half of hybrid retrieval runs
without re-indexing at retrieval time.

### How/when the DB is re-indexed

The `manifest.yaml` is cheap to recompute, so for every new run it is computed and
compared against the existing one. What happens next depends on what changed:

- **Incremental update (the common case)** — the source files changed but
  `chunk_size`, `chunk_overlap`, and `embedding_model` did not. In this case, only added
  and modified files are reloaded and embedded. Untouched files are never
  re-embedded, so editing one file in a large corpus costs one file's worth of work.
- **Full rebuild** — `chunk_size`, `chunk_overlap`, or `embedding_model` changed
  (every stored vector is then invalid), or there is no manifest yet. The persist
  directory is wiped, and everything is indexed from scratch.

The BM25 index has no incremental update path, so it is rebuilt after every update —
from the chunks already stored in Chroma, which needs no embedding calls and no
re-reading of source files. This should be very fast anyway.


## TODOs

- [x] Incremental indexing (only re-embed files that changed)
- [x] Hybrid retrieval, tuning config defaults
- [x] Support for popular LLM providers via API keys
- [x] Conversation memory in chat mode
- [x] Pydantic validation of config file
- [ ] UX/UI tuning of CLI (improved commands/statuses, etc)
- [ ] Performance optimizations (DB creation/update, pipeline execution)

If something cool is missing, feel free to open an issue or a PR.

## License

MIT
