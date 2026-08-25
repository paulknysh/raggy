import yaml
from langchain_core.documents import Document

from raggy import vectorstore


def test_annotate_line_numbers_sets_start_and_end_lines():
    content = "line one\nline two\nline three\nline four\nline five\n"

    split = Document(
        page_content="line three\nline four\n",
        metadata={"source": "doc.txt"},
    )

    vectorstore.annotate_line_numbers([split], content)

    assert split.metadata["start_line"] == 3
    assert split.metadata["end_line"] == 4


def test_annotate_line_numbers_counts_first_line_as_one():
    content = "single line with no trailing newline"

    split = Document(page_content="single line with", metadata={})

    vectorstore.annotate_line_numbers([split], content)

    assert split.metadata["start_line"] == 1
    assert split.metadata["end_line"] == 1


def test_annotate_line_numbers_falls_back_for_overlapping_chunks():
    content = "alpha\nbeta\ngamma\nalpha\nbeta\n"

    split = Document(page_content="alpha\nbeta\n", metadata={})

    vectorstore.annotate_line_numbers([split], content)

    assert split.metadata["start_line"] == 1
    assert split.metadata["end_line"] == 2


def test_should_annotate_lines_only_for_text_based_files():
    line_annotated = [".txt", ".md", ".markdown", ".html", ".htm"]
    for suffix in line_annotated:
        doc = Document(page_content="x", metadata={"source": f"/tmp/doc{suffix}"})
        assert vectorstore._should_annotate_lines(doc) is True, suffix

    locationless = [".pdf", ".pptx", ".docx", ".png", ".jpg", ".jpeg", ".bmp"]
    for suffix in locationless:
        doc = Document(page_content="x", metadata={"source": f"/tmp/doc{suffix}"})
        assert vectorstore._should_annotate_lines(doc) is False, suffix


def test_initialize_db_ingests_when_collection_empty(tmp_path, monkeypatch):
    calls = {"ingest": 0}
    source_file = tmp_path / "doc.txt"
    source_file.write_text("content", encoding="utf-8")

    class FakeCollection:
        def count(self):
            return 0

    class FakeVectorstore:
        _collection = FakeCollection()

    monkeypatch.setattr(vectorstore, "get_vectorstore", lambda *_: FakeVectorstore())

    def fake_ingest_document(**kwargs):
        calls["ingest"] += 1
        assert kwargs["sources"] == [str(source_file)]

    monkeypatch.setattr(vectorstore, "ingest_document", fake_ingest_document)

    result = vectorstore.initialize_db(
        persist_directory=str(tmp_path / "db"),
        embedding_model="embed-model",
        sources=[str(source_file)],
        chunk_size=500,
        chunk_overlap=50,
        n_batches=20,
    )

    assert isinstance(result, FakeVectorstore)
    assert calls["ingest"] == 1


def test_initialize_db_skips_ingest_when_data_exists(monkeypatch, tmp_path):
    calls = {"ingest": 0}
    source_file = tmp_path / "doc.txt"
    source_file.write_text("content", encoding="utf-8")

    class FakeCollection:
        def count(self):
            return 4

    class FakeVectorstore:
        _collection = FakeCollection()

    persist_directory = tmp_path / "db"
    persist_directory.mkdir()

    index_cfg = vectorstore.build_index_config(
        sources=[str(source_file)],
        chunk_size=500,
        chunk_overlap=50,
        embedding_model="embed-model",
    )
    (persist_directory / "manifest.yaml").write_text(
        yaml.safe_dump(index_cfg), encoding="utf-8"
    )

    monkeypatch.setattr(vectorstore, "get_vectorstore", lambda *_: FakeVectorstore())
    monkeypatch.setattr(
        vectorstore,
        "ingest_document",
        lambda **_: calls.__setitem__("ingest", calls["ingest"] + 1),
    )

    vectorstore.initialize_db(
        persist_directory=str(persist_directory),
        embedding_model="embed-model",
        sources=[str(source_file)],
        chunk_size=500,
        chunk_overlap=50,
        n_batches=20,
    )

    assert calls["ingest"] == 0


def test_initialize_db_rebuilds_when_manifest_differs(tmp_path, monkeypatch):
    calls = {"ingest": 0}
    source_file = tmp_path / "doc.txt"
    source_file.write_text("content", encoding="utf-8")

    class FakeCollection:
        def count(self):
            return 4

    class FakeVectorstore:
        _collection = FakeCollection()

        def reset_collection(self):
            return None

    manifest = tmp_path / "db" / "manifest.yaml"
    manifest.parent.mkdir()
    manifest.write_text(
        yaml.safe_dump(
            {
                "sources": [str(source_file)],
                "chunk_size": 999,
                "chunk_overlap": 50,
                "embedding_model": "embed-model",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(vectorstore, "get_vectorstore", lambda *_: FakeVectorstore())
    monkeypatch.setattr(
        vectorstore,
        "ingest_document",
        lambda **_: calls.__setitem__("ingest", calls["ingest"] + 1),
    )

    vectorstore.initialize_db(
        persist_directory=str(tmp_path / "db"),
        embedding_model="embed-model",
        sources=[str(source_file)],
        chunk_size=500,
        chunk_overlap=50,
        n_batches=20,
    )

    assert calls["ingest"] == 1
    assert (tmp_path / "db" / "manifest.yaml").read_text(
        encoding="utf-8"
    ) == yaml.safe_dump(
        {
            "sources": [str(source_file)],
            "chunk_size": 500,
            "chunk_overlap": 50,
            "embedding_model": "embed-model",
            "content_hash": vectorstore._file_fingerprint([str(source_file)]),
        }
    )
