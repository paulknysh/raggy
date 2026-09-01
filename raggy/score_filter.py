"""Relevance filtering driven by cross-encoder reranker scores.

The reranker returns a relevance probability in (0, 1) per (query, chunk)
pair (see :mod:`raggy.reranker`). The stock LangChain ``CrossEncoderReranker``
discards these scores once it has sorted and selected the top-k chunks, so to
threshold on them we use :class:`ScoreAnnotatingReranker`, which stamps each
surviving chunk with its score before returning, and then
:func:`filter_by_score_threshold` drops the low-scoring tail before the final
generation step.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict

SCORE_KEY = "relevance_score"


class ScoreAnnotatingReranker(BaseDocumentCompressor):
    """Cross-encoder reranker that keeps each chunk's relevance score.

    Behaves like LangChain's ``CrossEncoderReranker`` (score, sort descending,
    take the top ``top_n``) but additionally writes the score into
    ``doc.metadata[SCORE_KEY]`` for every returned chunk so a downstream
    threshold filter can act on it.
    """

    model: BaseCrossEncoder
    top_n: int = 3

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        """Score each (query, chunk) pair, annotate, keep the top ``top_n``."""
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        for doc, score in zip(documents, scores):
            doc.metadata[SCORE_KEY] = float(score)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]


def filter_by_score_threshold(
    docs: list[Document], threshold: float | None
) -> list[Document]:
    """Keep only chunks whose reranker score is at or above ``threshold``.

    Documents without a stored score (e.g. reranking was disabled) are kept,
    so this is fail-open. A ``threshold`` of ``None`` or ``<= 0`` disables the
    filter entirely and returns ``docs`` unchanged. Order is preserved.
    """
    if not threshold or threshold <= 0:
        return docs
    return [d for d in docs if d.metadata.get(SCORE_KEY, float("inf")) >= threshold]
