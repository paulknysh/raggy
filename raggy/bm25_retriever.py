"""BM25 lexical retriever backed by the ``bm25s`` index persisted at build time.

The dense vector store (Chroma) has no lexical understanding, so hybrid
retrieval pairs it with a sparse BM25 pass. The index is built and saved to
``<persist_directory>/bm25_index`` when the vector DB is constructed (see
:mod:`raggy.vectorstore`) and loaded here at retrieval time.
"""

import json
from pathlib import Path

import bm25s
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

BM25_INDEX_DIRNAME = "bm25_index"
METADATA_FILENAME = "chunks_metadata.json"


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
