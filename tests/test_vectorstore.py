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

    def fake_create_index(**kwargs):
        calls["ingest"] += 1
        assert kwargs["sources"] == [str(source_file)]

    monkeypatch.setattr(vectorstore, "create_index", fake_create_index)

    result = vectorstore.initialize_db(
        persist_directory=str(tmp_path / "db"),
        embedding_model="embed-model",
        sources=[str(source_file)],
        chunk_size=500,
        chunk_overlap=50,
        batch_size=100,
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
        "create_index",
        lambda **_: calls.__setitem__("ingest", calls["ingest"] + 1),
    )

    vectorstore.initialize_db(
        persist_directory=str(persist_directory),
        embedding_model="embed-model",
        sources=[str(source_file)],
        chunk_size=500,
        chunk_overlap=50,
        batch_size=100,
    )

    assert calls["ingest"] == 0


def test_save_bm25_index_persists_expected_files(tmp_path):
    from langchain_core.documents import Document

    from raggy.bm25_retriever import BM25_INDEX_DIRNAME, METADATA_FILENAME

    splits = [
        Document(page_content="chunk one", metadata={"source": "a.txt", "page": 1}),
        Document(page_content="chunk two", metadata={"source": "b.txt", "page": 2}),
    ]
    vectorstore._save_bm25_index(splits, str(tmp_path))

    index_dir = tmp_path / BM25_INDEX_DIRNAME
    assert index_dir.is_dir()
    assert (index_dir / METADATA_FILENAME).exists()
    assert (index_dir / "data.csc.index.npy").exists()


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
        "create_index",
        lambda **_: calls.__setitem__("ingest", calls["ingest"] + 1),
    )

    vectorstore.initialize_db(
        persist_directory=str(tmp_path / "db"),
        embedding_model="embed-model",
        sources=[str(source_file)],
        chunk_size=500,
        chunk_overlap=50,
        batch_size=100,
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
            "files": vectorstore.file_fingerprints([str(source_file)]),
        }
    )


def _write_manifest_for(persist_directory, source_file, **overrides):
    index_cfg = vectorstore.build_index_config(
        sources=[str(source_file)],
        chunk_size=500,
        chunk_overlap=50,
        embedding_model="embed-model",
    )
    index_cfg.update(overrides)
    persist_directory.mkdir(parents=True, exist_ok=True)
    (persist_directory / "manifest.yaml").write_text(
        yaml.safe_dump(index_cfg), encoding="utf-8"
    )
    return index_cfg


def test_plan_index_update_requires_full_rebuild_without_manifest(tmp_path):
    plan = vectorstore.plan_index_update(str(tmp_path / "db"), {"files": {}})

    assert plan.full_rebuild is True
    assert plan.has_changes is True


def test_plan_index_update_full_rebuild_on_chunking_change(tmp_path):
    source_file = tmp_path / "doc.txt"
    source_file.write_text("content", encoding="utf-8")
    persist_directory = tmp_path / "db"
    _write_manifest_for(persist_directory, source_file, chunk_size=999)

    index_cfg = vectorstore.build_index_config(
        sources=[str(source_file)],
        chunk_size=500,
        chunk_overlap=50,
        embedding_model="embed-model",
    )
    plan = vectorstore.plan_index_update(str(persist_directory), index_cfg)

    assert plan.full_rebuild is True


def test_plan_index_update_full_rebuild_for_legacy_manifest(tmp_path):
    source_file = tmp_path / "doc.txt"
    source_file.write_text("content", encoding="utf-8")
    persist_directory = tmp_path / "db"
    persist_directory.mkdir()
    (persist_directory / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "sources": [str(source_file)],
                "chunk_size": 500,
                "chunk_overlap": 50,
                "embedding_model": "embed-model",
                "content_hash": "deadbeef",
            }
        ),
        encoding="utf-8",
    )

    index_cfg = vectorstore.build_index_config(
        sources=[str(source_file)],
        chunk_size=500,
        chunk_overlap=50,
        embedding_model="embed-model",
    )
    plan = vectorstore.plan_index_update(str(persist_directory), index_cfg)

    assert plan.full_rebuild is True


def test_plan_index_update_detects_added_modified_and_removed(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    kept = docs_dir / "kept.txt"
    changed = docs_dir / "changed.txt"
    removed = docs_dir / "removed.txt"
    for path in (kept, changed, removed):
        path.write_text("original", encoding="utf-8")

    persist_directory = tmp_path / "db"
    persist_directory.mkdir()
    stored = vectorstore.build_index_config(
        sources=[str(docs_dir)],
        chunk_size=500,
        chunk_overlap=50,
        embedding_model="embed-model",
    )
    (persist_directory / "manifest.yaml").write_text(
        yaml.safe_dump(stored), encoding="utf-8"
    )

    changed.write_text("edited", encoding="utf-8")
    removed.unlink()
    added = docs_dir / "added.txt"
    added.write_text("brand new", encoding="utf-8")

    index_cfg = vectorstore.build_index_config(
        sources=[str(docs_dir)],
        chunk_size=500,
        chunk_overlap=50,
        embedding_model="embed-model",
    )
    plan = vectorstore.plan_index_update(str(persist_directory), index_cfg)

    assert plan.full_rebuild is False
    assert plan.added == [str(added)]
    assert plan.modified == [str(changed)]
    assert plan.removed == [str(removed)]
    assert str(kept) not in plan.added + plan.modified + plan.removed


def test_plan_index_update_reports_no_changes_when_sources_are_untouched(tmp_path):
    source_file = tmp_path / "doc.txt"
    source_file.write_text("content", encoding="utf-8")
    persist_directory = tmp_path / "db"
    index_cfg = _write_manifest_for(persist_directory, source_file)

    plan = vectorstore.plan_index_update(str(persist_directory), index_cfg)

    assert plan.has_changes is False
    assert vectorstore.db_needs_rebuild(str(persist_directory), index_cfg) is False


def test_update_index_deletes_stale_chunks_and_embeds_changed_files(
    tmp_path, monkeypatch
):
    from langchain_core.documents import Document

    deleted: list = []
    embedded: list = []
    saved: list = []

    class FakeCollection:
        def delete(self, where):
            deleted.append(where)

        def get(self, include, limit=None, offset=0):
            return {
                "documents": ["kept chunk"][offset : offset + (limit or 1)],
                "metadatas": [{"source": "kept.txt"}][offset : offset + (limit or 1)],
            }

    class FakeVectorstore:
        _collection = FakeCollection()

    monkeypatch.setattr(
        vectorstore,
        "load_documents_from_paths",
        lambda paths: [
            Document(page_content="new text", metadata={"source": str(paths[0])})
        ],
    )
    monkeypatch.setattr(
        vectorstore,
        "_embed_in_batches",
        lambda splits, store, batch_size: embedded.extend(splits),
    )
    monkeypatch.setattr(
        vectorstore,
        "_save_bm25_index",
        lambda splits, persist_directory: saved.append(splits),
    )

    plan = vectorstore.IndexPlan(added=["a.txt"], modified=["b.txt"], removed=["c.txt"])
    vectorstore.update_index(
        vectorstore=FakeVectorstore(),
        plan=plan,
        chunk_size=500,
        chunk_overlap=50,
        batch_size=100,
        persist_directory=str(tmp_path),
    )

    assert deleted == [{"source": {"$in": ["c.txt", "b.txt"]}}]
    assert [doc.page_content for doc in embedded] == ["new text"]
    # BM25 is rebuilt from what Chroma holds after the update, not from the
    # newly embedded chunks alone.
    assert [doc.page_content for doc in saved[0]] == ["kept chunk"]


def test_initialize_db_updates_incrementally_when_only_sources_change(
    tmp_path, monkeypatch
):
    calls = {"ingest": 0, "update": 0}
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    doc = docs_dir / "doc.txt"
    doc.write_text("content", encoding="utf-8")

    persist_directory = tmp_path / "db"
    persist_directory.mkdir()
    stored = vectorstore.build_index_config(
        sources=[str(docs_dir)],
        chunk_size=500,
        chunk_overlap=50,
        embedding_model="embed-model",
    )
    (persist_directory / "manifest.yaml").write_text(
        yaml.safe_dump(stored), encoding="utf-8"
    )

    added = docs_dir / "added.txt"
    added.write_text("brand new", encoding="utf-8")

    class FakeCollection:
        def count(self):
            return 4

    class FakeVectorstore:
        _collection = FakeCollection()

    monkeypatch.setattr(vectorstore, "get_vectorstore", lambda *_: FakeVectorstore())
    monkeypatch.setattr(
        vectorstore,
        "create_index",
        lambda **_: calls.__setitem__("ingest", calls["ingest"] + 1),
    )

    def fake_update_index(**kwargs):
        calls["update"] += 1
        assert kwargs["plan"].added == [str(added)]

    monkeypatch.setattr(vectorstore, "update_index", fake_update_index)
    wiped: list = []
    monkeypatch.setattr(
        vectorstore, "_reset_persist_directory", lambda path: wiped.append(path)
    )

    vectorstore.initialize_db(
        persist_directory=str(persist_directory),
        embedding_model="embed-model",
        sources=[str(docs_dir)],
        chunk_size=500,
        chunk_overlap=50,
        batch_size=100,
    )

    assert calls == {"ingest": 0, "update": 1}
    assert wiped == []
    manifest = yaml.safe_load(
        (persist_directory / "manifest.yaml").read_text(encoding="utf-8")
    )
    assert str(added) in manifest["files"]


def test_save_bm25_index_removes_index_for_empty_corpus(tmp_path):
    from langchain_core.documents import Document

    from raggy.bm25_retriever import BM25_INDEX_DIRNAME

    vectorstore._save_bm25_index(
        [Document(page_content="chunk", metadata={})], str(tmp_path)
    )
    assert (tmp_path / BM25_INDEX_DIRNAME).is_dir()

    vectorstore._save_bm25_index([], str(tmp_path))
    assert not (tmp_path / BM25_INDEX_DIRNAME).exists()


def test_collection_chunks_pages_through_large_collections(monkeypatch):
    monkeypatch.setattr(vectorstore, "_COLLECTION_PAGE_SIZE", 10)
    total = 25
    pages: list[tuple[int, int]] = []

    class FakeCollection:
        def get(self, include, limit=None, offset=0):
            pages.append((limit, offset))
            end = min(offset + limit, total)
            return {
                "documents": [f"chunk {i}" for i in range(offset, end)],
                "metadatas": [{"source": f"doc{i}.txt"} for i in range(offset, end)],
            }

    class FakeVectorstore:
        _collection = FakeCollection()

    chunks = vectorstore._collection_chunks(FakeVectorstore())

    assert [doc.page_content for doc in chunks] == [f"chunk {i}" for i in range(total)]
    assert chunks[-1].metadata == {"source": "doc24.txt"}
    # Three requests: two full pages plus a short final page that ends the loop.
    assert pages == [(10, 0), (10, 10), (10, 20)]


def test_collection_chunks_handles_an_empty_collection():
    class FakeCollection:
        def get(self, include, limit=None, offset=0):
            return {"documents": [], "metadatas": []}

    class FakeVectorstore:
        _collection = FakeCollection()

    assert vectorstore._collection_chunks(FakeVectorstore()) == []
