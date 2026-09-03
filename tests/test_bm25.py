from langchain_core.documents import Document

from raggy.bm25 import (
    BM25_INDEX_DIRNAME,
    METADATA_FILENAME,
    get_bm25_retriever,
    save_bm25_index,
)


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
    save_bm25_index(splits, str(tmp_path))

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
    save_bm25_index(splits, str(tmp_path))

    retriever = get_bm25_retriever(str(tmp_path), k=1)
    docs = retriever.invoke("zzzz nonexistent")
    assert len(docs) == 1


def test_save_bm25_index_persists_expected_files(tmp_path):
    splits = [
        Document(page_content="chunk one", metadata={"source": "a.txt", "page": 1}),
        Document(page_content="chunk two", metadata={"source": "b.txt", "page": 2}),
    ]
    save_bm25_index(splits, str(tmp_path))

    index_dir = tmp_path / BM25_INDEX_DIRNAME
    assert index_dir.is_dir()
    assert (index_dir / METADATA_FILENAME).exists()
    assert (index_dir / "data.csc.index.npy").exists()


def test_save_bm25_index_removes_index_for_empty_corpus(tmp_path):
    save_bm25_index([Document(page_content="chunk", metadata={})], str(tmp_path))
    assert (tmp_path / BM25_INDEX_DIRNAME).is_dir()

    save_bm25_index([], str(tmp_path))
    assert not (tmp_path / BM25_INDEX_DIRNAME).exists()
