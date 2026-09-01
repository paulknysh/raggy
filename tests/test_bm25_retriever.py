from langchain_core.documents import Document

from raggy import vectorstore
from raggy.bm25_retriever import get_bm25_retriever


def test_bm25_retriever_round_trips_documents(tmp_path):
    splits = [
        Document(
            page_content="the quick brown fox",
            metadata={"source": "a.txt", "page": 1},
        ),
        Document(
            page_content="jumps over the lazy dog",
            metadata={"source": "b.txt", "page": 2},
        ),
    ]
    vectorstore._save_bm25_index(splits, str(tmp_path))

    retriever = get_bm25_retriever(str(tmp_path), k=1)
    docs = retriever.invoke("quick dog")

    assert len(docs) == 1
    assert docs[0].metadata == {"source": "a.txt", "page": 1}


def test_get_bm25_retriever_raises_when_index_missing(tmp_path):
    try:
        get_bm25_retriever(str(tmp_path))
    except FileNotFoundError as exc:
        assert str(tmp_path) in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_bm25_retriever_handles_unseen_tokens(tmp_path):
    splits = [Document(page_content="the quick brown fox", metadata={})]
    vectorstore._save_bm25_index(splits, str(tmp_path))

    retriever = get_bm25_retriever(str(tmp_path), k=1)
    docs = retriever.invoke("zzzz nonexistent")
    assert len(docs) == 1
