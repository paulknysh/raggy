# raggy

A lightweight CLI tool for Retrieval-Augmented Generation (RAG) over local documents built with LangChain, Chroma, and Ollama. Hybrid database (vector + BM25 index) and embedding generation run fully locally. Answer generation can run either via a local LLM or remotely using an API key. `raggy` supports most common document formats and handles images/scans automatically via OCR.

Usage example -- CLI returns an answer based on your documents, and citations along with their locations and relevance scores:

<img src="assets/cli_demo.png">

These are all currently supported file formats (all other formats are ignored):

| Type | Extensions |
| --- | --- |
| Documents | `.pdf`, `.docx`, `.pptx` |
| Text | `.txt`, `.md`, `.markdown` |
| Web | `.html`, `.htm` |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.bmp` |

## Prerequisites

Ollama is required for running the local embedding model (which feeds the on-disk vector DB), and also a local LLM (if needed). To install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh

# may need to start ollama after installation using the app or:
ollama
# or
ollama serve
```

If an API key will be used for accessing an LLM remotely, a standard environment variable needs to be set (one of the following):

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

Then install using:

```bash
# with uv
uv tool install -e .

# with pipx
pipx install -e .
```

For now, cloning + editable install is picked as a preferred installation method, as it allows you to experiment with the demo dataset, run the eval harness, and edit/debug code if needed. In the future, direct install via `uv tool install git+https ...`/`pipx install git+https ...` will be used instead.

## Usage (CLI)

First, run this command:

```bash
make config
```

It creates your own user config (`config.yaml`) where all your execution parameters live. For a detailed overview of all config parameters, see [Configuration](#configuration). While `config.yaml` comes with defaults you can test, you should populate `sources` (your input folders/files) and `db_directory` (DB location) sections with your preferred paths.

To start the CLI, use the `raggy <path-to-config-file>` command:

```bash
raggy config.yaml
```

> [!IMPORTANT]
> CLI automatically pulls all models listed in `config.yaml` and (re-)indexes your documents -- this might take a while on the first run, depending on models chosen, document count/size, and whether OCR is needed (scans, images, etc).

> [!IMPORTANT]
> Relative paths in `config.yaml` resolve against the current directory (from where `raggy` command is executed). Keep that in mind if you want to run `raggy` from other locations. To be safe, just always use absolute paths in your config file.

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

All runtime settings are defined in the config file:

| Setting | Description |
| --- | --- |
| `sources` | list of source directories and/or files |
| `db_directory` | location where the DB itself is stored |
| `embedding_model` | Ollama embedding model (e.g. `nomic-embed-text`) |
| `chunk_size` | chunk size in characters |
| `chunk_overlap` | character overlap between adjacent chunks |
| `embed_batch_size` | max number of chunks embedded per batch into Chroma (`100` in the shipped config); the number of batches is derived automatically |
| `llm_provider` | where generation runs: `ollama` (local, the shipped value) or `openai`/`anthropic`/`google` (via API) |
| `llm_model` | chat model for generation (e.g. `phi4-mini` locally, or a remote model name like `gemini-3.7-flash`) |
| `llm_temperature` | LLM sampling temperature |
| `retrieve_k` | total chunks retrieved per query, split across the dense and lexical retrievals (`50` in the shipped config) |
| `hybrid_alpha` | fraction of `retrieve_k` spent on the vector retrieval; the remainder goes to lexical (`1.0` = vector only, `0.0` = lexical only, `0.5` in the shipped config) |
| `rerank_model` | Hugging Face ID of the cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L6-v2`) |
| `rerank_k` | number of chunks returned by the cross-encoder (must be `<= retrieve_k`) |
| `rerank_threshold` | drops reranked chunks whose relevance score is below this value (`0.0` = disabled, `0.3` in the shipped config) |
| `system_prompt` | system prompt dictating how the LLM should answer; must contain a `{context}` placeholder |

Notes:

- The current default config parameters were tested on a basic MacBook Air with 8GB RAM. Switching to much heavier local models likely needs appropriate GPU/memory.

- Chroma doesn't seem to be able to embed all chunks in one go; therefore, `embed_batch_size` was introduced so it's done in batches instead. 100 seems like a reasonable default, but if you get Chroma errors during embedding (such as `Error: Post "http://127.0.0.1:50175/tokenize": EOF (status code: 400)`), try lowering `embed_batch_size` further.

## Pipeline

Below are the main steps in the RAG pipeline (assuming the DB is already created):

**[1] Hybrid retrieval.** `retrieve_k` is a total *candidate budget*, split by
`hybrid_alpha` between two retrievals over the whole corpus:

- **dense** retrieval -- nearest chunks in Chroma by embedding similarity (good at
  paraphrase and synonyms);
- **lexical** retrieval -- the persisted `bm25s` index (good at exact terms:
  identifiers, names, acronyms, numbers).

So `retrieve_k: 50` with `hybrid_alpha: 0.5` takes 25 chunks from each arm. The two
ranked lists are merged by **reciprocal rank fusion**, which collapses duplicates and
needs no score calibration between the two very different scales. Fusion weights are
uniform on purpose: `hybrid_alpha` already sets each arm's influence by deciding how
many candidates it contributes.

**[2] Cross-encoder reranking.** Stage 1 favors recall and is noisy. The reranker
(`rerank_model`, run locally on onnxruntime) pushes the query and the chunk through
the model *together* and emits one relevance score per pair -- far sharper than
cosine distance between separately embedded texts, and affordable on ~50 chunks
though not on the whole corpus. The top `rerank_k` chunks survive.

**[3] Score threshold.** Each chunk carries its reranker score in
`doc.metadata["relevance_score"]`, and anything below `rerank_threshold` is dropped.
This keeps `rerank_k` from polluting the context when the corpus has no good answer;
`0.0` disables it.

**[4] Generation.** The survivors are concatenated into `{context}` in
`system_prompt` and sent to the configured LLM, along with the chat history (in chat
mode) and the question. The exact chunks the model saw are returned to the caller --
`retrieved_docs` above, and the source table the CLI prints.

## The DB mechanics

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

## Demo dataset

This article (https://arxiv.org/abs/2608.06223v1) is used here as a demo dataset. It's an 8-page document -- each page is saved in different file formats (including PDF, plaintext, images, MS Office) and saved inside the `sample_docs` directory. This directory is specified in `config.yaml` by default.

## Demo eval

`eval` folder currently contains a basic harness to test pipeline performance on the demo dataset. You can run it by:

```bash
uv run eval/run_eval.py
```

It computes basic retrieval/generation metrics and produces a summary (both printed and saved to `eval/results.json`). The Q&A pairs are about `sample_docs`, so the harness always runs against `default_config/default_config.yaml` rather than your own `config.yaml`.

## TODOs

- [x] Incremental indexing (only re-embed files that changed)
- [x] Hybrid retrieval, tuning config defaults
- [x] Support for popular LLM providers via API keys
- [x] Conversation memory in chat mode
- [x] Pydantic validation of config file
- [ ] UX/UI tuning of CLI (improved commands/statuses, etc)
- [ ] Performance optimizations (DB creation/update, pipeline execution)

If some features are not working or missing, feel free to open an issue or a PR.

## License

MIT
