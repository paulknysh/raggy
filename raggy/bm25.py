"""BM25 lexical retrieval: building the persisted ``bm25s`` index and reading it.

The dense vector store (Chroma) has no lexical understanding, so hybrid
retrieval pairs it with a sparse BM25 pass. :func:`save_bm25_index` writes the
index to ``<persist_directory>/bm25_index`` when the vector DB is built (see
:mod:`raggy.indexing`); :func:`get_bm25_retriever` loads it back at
retrieval time.
"""

import json
import shutil
from pathlib import Path

import bm25s
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

BM25_INDEX_DIRNAME = "bm25_index"
METADATA_FILENAME = "chunks_metadata.json"


def save_bm25_index(splits: list[Document], persist_directory: str) -> None:
    """Build and persist a ``bm25s`` index (plus chunk metadata) to disk.

    The index is written to ``<persist_directory>/bm25_index`` at DB build
    time so the retrieval step can load it later without re-indexing. Corpus
    entries are aligned by index with the per-chunk metadata so the
    ``Bm25sRetriever`` can reconstruct the original ``Document``s.
    """
    index_dir = Path(persist_directory) / BM25_INDEX_DIRNAME
    if not splits:
        # An empty corpus can't be indexed; drop the old index rather than
        # leaving one that would keep returning removed chunks.
        shutil.rmtree(index_dir, ignore_errors=True)
        return
    corpus = [split.page_content for split in splits]
    bm25 = bm25s.BM25()
    bm25.index(bm25s.tokenize(corpus, show_progress=False), show_progress=False)
    index_dir.mkdir(parents=True, exist_ok=True)
    bm25.save(str(index_dir), corpus=corpus, show_progress=False)
    metadata = [dict(split.metadata) for split in splits]
    (index_dir / METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")


class Bm25sRetriever(BaseRetriever):
    """A LangChain retriever wrapping a persisted ``bm25s`` index.

    Loads the BM25 index and the per-chunk metadata saved alongside it during
    building, and returns ``Document`` objects whose ``page_content`` and
    ``metadata`` match the original indexed chunks.
    """

    bm25: bm25s.BM25
    chunks_metadata: list[dict]
    k: int = 10

    def _get_relevant_documents(self, query: str) -> list[Document]:
        tokenized = bm25s.tokenize([query], show_progress=False)
        hits, _ = self.bm25.retrieve(
            tokenized, corpus=self.bm25.corpus, k=self.k, show_progress=False
        )
        docs: list[Document] = []
        for entry in hits[0]:
            idx = int(entry["id"])
            docs.append(
                Document(
                    page_content=entry["text"],
                    metadata=dict(self.chunks_metadata[idx]),
                )
            )
        return docs


def get_bm25_retriever(persist_directory: str, k: int = 10) -> Bm25sRetriever:
    """Load the persisted ``bm25s`` index from ``persist_directory``.

    Raises ``FileNotFoundError`` if the index was never built (e.g. hybrid
    search enabled without ever running the DB build step).
    """
    index_dir = Path(persist_directory) / BM25_INDEX_DIRNAME
    metadata_path = index_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found in '{index_dir}'. Rebuild the DB with "
            "hybrid_search enabled before using it."
        )

    bm25 = bm25s.BM25()
    bm25 = bm25.load(str(index_dir), load_corpus=True, show_progress=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return Bm25sRetriever(bm25=bm25, chunks_metadata=metadata, k=k)
