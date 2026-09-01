from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document

from raggy import score_filter
from raggy.score_filter import SCORE_KEY


class FakeModel(BaseCrossEncoder):
    def __init__(self, scores):
        self.scores = scores

    def score(self, pairs):
        return self.scores


def _docs(*contents):
    return [Document(page_content=c) for c in contents]


def test_reranker_annotates_scores_and_keeps_top_n():
    model = FakeModel([0.1, 0.9, 0.5])
    compressor = score_filter.ScoreAnnotatingReranker(model=model, top_n=2)

    kept = compressor.compress_documents(_docs("A", "B", "C"), "q")

    assert [d.page_content for d in kept] == ["B", "C"]
    assert kept[0].metadata[SCORE_KEY] == 0.9
    assert kept[1].metadata[SCORE_KEY] == 0.5


def test_reranker_sorting_is_descending_by_score():
    model = FakeModel([0.2, 0.8, 0.6])
    compressor = score_filter.ScoreAnnotatingReranker(model=model, top_n=3)

    kept = compressor.compress_documents(_docs("A", "B", "C"), "q")

    assert [d.page_content for d in kept] == ["B", "C", "A"]


def test_threshold_drops_low_scoring_keep_order():
    docs = _docs("A", "B", "C")
    docs[0].metadata[SCORE_KEY] = 0.8
    docs[1].metadata[SCORE_KEY] = 0.2
    docs[2].metadata[SCORE_KEY] = 0.5

    kept = score_filter.filter_by_score_threshold(docs, 0.3)

    assert [d.page_content for d in kept] == ["A", "C"]


def test_threshold_disabled_when_zero_or_none():
    docs = _docs("A", "B")
    docs[0].metadata[SCORE_KEY] = 0.1

    assert score_filter.filter_by_score_threshold(docs, 0.0) == docs
    assert score_filter.filter_by_score_threshold(docs, None) == docs


def test_threshold_fail_open_for_unscored_docs():
    docs = _docs("A", "B")
    docs[0].metadata[SCORE_KEY] = 0.8
    # B has no score -> kept

    kept = score_filter.filter_by_score_threshold(docs, 0.5)

    assert [d.page_content for d in kept] == ["A", "B"]
