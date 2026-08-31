from types import SimpleNamespace

import pytest

from raggy import raggy


def test_load_config_raises_for_missing_file(tmp_path):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        raggy.load_config(str(missing))


def test_load_config_reads_all_values(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources:\n"
        "  - ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 900\n"
        "chunk_overlap: 600\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_model: test-llm\n"
        "temperature: 0.4\n"
        "retrieve_k: 11\n"
        "mmr_fetch_k: 22\n"
        "search_type: similarity\n"
        "relevance_filter: true\n"
        "rerank_enabled: true\n"
        "rerank_model: rerank-x\n"
        "rerank_k: 4\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    cfg = raggy.load_config(str(config_file))

    assert cfg == {
        "sources": ["./docs"],
        "persist_directory": "./my_db",
        "chunk_size": 900,
        "chunk_overlap": 600,
        "batch_size": 100,
        "embedding_model": "test-embed",
        "llm_provider": "ollama",
        "llm_model": "test-llm",
        "temperature": 0.4,
        "retrieve_k": 11,
        "mmr_fetch_k": 22,
        "search_type": "similarity",
        "relevance_filter": True,
        "rerank_enabled": True,
        "rerank_model": "rerank-x",
        "rerank_k": 4,
        "system_prompt": "use this",
    }


def test_load_config_llm_provider_defaults_to_ollama(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources:\n"
        "  - ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 500\n"
        "chunk_overlap: 100\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_model: test-llm\n"
        "temperature: 0.0\n"
        "retrieve_k: 5\n"
        "mmr_fetch_k: 25\n"
        "search_type: mmr\n"
        "rerank_enabled: false\n"
        "rerank_model: rerank-x\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    cfg = raggy.load_config(str(config_file))

    assert cfg["llm_provider"] == "ollama"
    assert cfg["relevance_filter"] is False


def test_load_config_reads_relevance_filter_false(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources:\n"
        "  - ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 500\n"
        "chunk_overlap: 100\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_model: test-llm\n"
        "temperature: 0.0\n"
        "retrieve_k: 5\n"
        "mmr_fetch_k: 25\n"
        "search_type: mmr\n"
        "relevance_filter: false\n"
        "rerank_enabled: false\n"
        "rerank_model: rerank-x\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    cfg = raggy.load_config(str(config_file))

    assert cfg["relevance_filter"] is False


def test_load_config_reads_llm_provider(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources:\n"
        "  - ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 500\n"
        "chunk_overlap: 100\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_provider: google\n"
        "llm_model: test-llm\n"
        "temperature: 0.0\n"
        "retrieve_k: 5\n"
        "mmr_fetch_k: 25\n"
        "search_type: mmr\n"
        "rerank_enabled: false\n"
        "rerank_model: rerank-x\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    cfg = raggy.load_config(str(config_file))

    assert cfg["llm_provider"] == "google"


def test_load_config_rerank_k_defaults_to_retrieve_k(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources:\n"
        "  - ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 500\n"
        "chunk_overlap: 100\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_model: test-llm\n"
        "temperature: 0.0\n"
        "retrieve_k: 5\n"
        "mmr_fetch_k: 25\n"
        "search_type: mmr\n"
        "rerank_enabled: false\n"
        "rerank_model: rerank-x\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    cfg = raggy.load_config(str(config_file))

    assert cfg["rerank_k"] == 5


def test_load_config_rejects_rerank_k_greater_than_k(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources:\n"
        "  - ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 500\n"
        "chunk_overlap: 100\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_model: test-llm\n"
        "temperature: 0.0\n"
        "retrieve_k: 5\n"
        "mmr_fetch_k: 25\n"
        "search_type: mmr\n"
        "rerank_enabled: true\n"
        "rerank_model: rerank-x\n"
        "rerank_k: 6\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rerank_k"):
        raggy.load_config(str(config_file))


def test_load_config_accepts_single_string_source(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources: ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 500\n"
        "chunk_overlap: 100\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_model: test-llm\n"
        "temperature: 0.0\n"
        "retrieve_k: 5\n"
        "mmr_fetch_k: 25\n"
        "search_type: mmr\n"
        "rerank_enabled: false\n"
        "rerank_model: rerank-x\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    cfg = raggy.load_config(str(config_file))

    assert cfg["sources"] == ["./docs"]


def test_load_config_rejects_empty_sources(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources: []\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 500\n"
        "chunk_overlap: 100\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_model: test-llm\n"
        "temperature: 0.0\n"
        "retrieve_k: 5\n"
        "mmr_fetch_k: 25\n"
        "search_type: mmr\n"
        "rerank_enabled: false\n"
        "rerank_model: rerank-x\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sources"):
        raggy.load_config(str(config_file))


def test_load_config_rejects_unknown_llm_provider(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources: ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 500\n"
        "chunk_overlap: 100\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_provider: not-a-provider\n"
        "llm_model: test-llm\n"
        "temperature: 0.0\n"
        "retrieve_k: 5\n"
        "mmr_fetch_k: 25\n"
        "search_type: mmr\n"
        "rerank_enabled: false\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="llm_provider"):
        raggy.load_config(str(config_file))


def test_load_config_rejects_unknown_search_type(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources: ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 500\n"
        "chunk_overlap: 100\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_model: test-llm\n"
        "temperature: 0.0\n"
        "retrieve_k: 5\n"
        "search_type: bogus\n"
        "rerank_enabled: false\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="search_type"):
        raggy.load_config(str(config_file))


def test_load_config_similarity_without_mmr_fetch_k(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources: ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 500\n"
        "chunk_overlap: 100\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_model: test-llm\n"
        "temperature: 0.0\n"
        "retrieve_k: 5\n"
        "search_type: similarity\n"
        "rerank_enabled: false\n"
        "rerank_model: rerank-x\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    cfg = raggy.load_config(str(config_file))
    assert cfg["mmr_fetch_k"] is None
    assert cfg["search_type"] == "similarity"


def test_load_config_requires_mmr_fetch_k_for_mmr(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources: ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 500\n"
        "chunk_overlap: 100\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_model: test-llm\n"
        "temperature: 0.0\n"
        "retrieve_k: 5\n"
        "search_type: mmr\n"
        "rerank_enabled: false\n"
        "rerank_model: rerank-x\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mmr_fetch_k is required"):
        raggy.load_config(str(config_file))


def test_load_config_rejects_non_positive_chunk_size(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources: ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 0\n"
        "chunk_overlap: 100\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_model: test-llm\n"
        "temperature: 0.0\n"
        "retrieve_k: 5\n"
        "mmr_fetch_k: 25\n"
        "search_type: mmr\n"
        "rerank_enabled: false\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chunk_size"):
        raggy.load_config(str(config_file))


def test_load_config_rejects_negative_temperature(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources: ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 500\n"
        "chunk_overlap: 100\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_model: test-llm\n"
        "temperature: -0.1\n"
        "retrieve_k: 5\n"
        "mmr_fetch_k: 25\n"
        "search_type: mmr\n"
        "rerank_enabled: false\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="temperature"):
        raggy.load_config(str(config_file))


def test_load_config_rejects_mmr_fetch_k_less_than_retrieve_k(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources: ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: 500\n"
        "chunk_overlap: 100\n"
        "batch_size: 100\n"
        "embedding_model: test-embed\n"
        "llm_model: test-llm\n"
        "temperature: 0.0\n"
        "retrieve_k: 5\n"
        "mmr_fetch_k: 3\n"
        "search_type: mmr\n"
        "rerank_enabled: false\n"
        "rerank_model: rerank-x\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mmr_fetch_k"):
        raggy.load_config(str(config_file))


def test_load_config_coerces_numeric_strings(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "sources: ./docs\n"
        "persist_directory: ./my_db\n"
        "chunk_size: '500'\n"
        "chunk_overlap: '100'\n"
        "batch_size: '100'\n"
        "embedding_model: test-embed\n"
        "llm_model: test-llm\n"
        "temperature: '0.4'\n"
        "retrieve_k: '5'\n"
        "mmr_fetch_k: '25'\n"
        "search_type: mmr\n"
        "rerank_enabled: false\n"
        "rerank_model: rerank-x\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    cfg = raggy.load_config(str(config_file))

    assert cfg["chunk_size"] == 500
    assert cfg["temperature"] == 0.4
    assert cfg["retrieve_k"] == 5


def test_init_db_forwards_config_to_initialize_db(monkeypatch):
    fake_cfg = {
        "sources": ["./docs"],
        "persist_directory": "./persist",
        "chunk_size": 100,
        "chunk_overlap": 10,
        "batch_size": 100,
        "embedding_model": "embed-x",
    }
    captured = {}
    sentinel = object()

    monkeypatch.setattr(raggy, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(raggy, "ensure_ollama_model", lambda model: False)

    def fake_initialize_db(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(raggy, "initialize_db", fake_initialize_db)

    result = raggy._init_db()

    assert result is sentinel
    assert captured == {
        "persist_directory": "./persist",
        "embedding_model": "embed-x",
        "sources": ["./docs"],
        "chunk_size": 100,
        "chunk_overlap": 10,
        "batch_size": 100,
    }


def test_run_pipeline_returns_response_and_retrieved_docs(monkeypatch):
    fake_cfg = {
        "llm_model": "llm-x",
        "llm_provider": "ollama",
        "system_prompt": "sys",
        "search_type": "mmr",
        "retrieve_k": 2,
        "mmr_fetch_k": 10,
        "temperature": 0.0,
        "relevance_filter": True,
        "rerank_enabled": False,
        "rerank_model": "rerank-x",
        "rerank_k": 2,
    }

    captured = {}

    class FakeChain:
        def invoke(self, inputs):
            captured["input"] = inputs
            return f"answer:{inputs['question']}"

    def fake_build_rag_chain(**kwargs):
        captured["chat_history"] = kwargs["chat_history"]
        sink = kwargs["doc_sink"]
        sink.append(
            [
                SimpleNamespace(page_content="doc1"),
                SimpleNamespace(page_content="doc2"),
            ]
        )
        return FakeChain(), SimpleNamespace(invoke=lambda q: [])

    monkeypatch.setattr(raggy, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(raggy, "_get_vectorstore", lambda: object())
    monkeypatch.setattr(raggy, "ensure_ollama_model", lambda model: False)
    monkeypatch.setattr(raggy, "build_rag_chain", fake_build_rag_chain)

    response, docs = raggy.run_pipeline("hello")

    assert response == "answer:hello"
    assert [doc.page_content for doc in docs] == ["doc1", "doc2"]
    assert captured["chat_history"] is None
    assert captured["input"] == {"question": "hello", "chat_history": []}


def test_run_pipeline_forwards_chat_history(monkeypatch):
    fake_cfg = {
        "llm_model": "llm-x",
        "llm_provider": "ollama",
        "system_prompt": "sys",
        "search_type": "mmr",
        "retrieve_k": 2,
        "mmr_fetch_k": 10,
        "temperature": 0.0,
        "relevance_filter": True,
        "rerank_enabled": False,
        "rerank_model": "rerank-x",
        "rerank_k": 2,
    }
    captured = {}
    history = [("human", "hello"), ("ai", "hi")]

    class FakeChain:
        def invoke(self, inputs):
            captured["input"] = inputs
            return "answer"

    def fake_build_rag_chain(**kwargs):
        captured["chat_history"] = kwargs["chat_history"]
        return FakeChain(), SimpleNamespace(invoke=lambda q: [])

    monkeypatch.setattr(raggy, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(raggy, "_get_vectorstore", lambda: object())
    monkeypatch.setattr(raggy, "ensure_ollama_model", lambda model: False)
    monkeypatch.setattr(raggy, "build_rag_chain", fake_build_rag_chain)

    raggy.run_pipeline("follow-up", chat_history=history)

    assert captured["chat_history"] is history
    assert captured["input"] == {"question": "follow-up", "chat_history": history}


def test_refresh_db_rebuilds_when_stale(monkeypatch):
    fake_cfg = {
        "sources": ["./docs"],
        "persist_directory": "./persist",
        "chunk_size": 100,
        "chunk_overlap": 10,
        "embedding_model": "embed-x",
    }
    old_store = object()
    new_store = object()
    calls = {"close": 0, "init": 0}

    monkeypatch.setattr(raggy, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(raggy, "build_index_config", lambda **kwargs: {"fake": True})
    monkeypatch.setattr(raggy, "db_needs_rebuild", lambda *_: True)
    monkeypatch.setattr(
        raggy,
        "close_vectorstore",
        lambda store: calls.__setitem__("close", calls["close"] + 1),
    )
    monkeypatch.setattr(raggy, "_init_db", lambda: new_store)
    raggy._vectorstore = old_store

    try:
        rebuilt = raggy.refresh_db()
        store = raggy._vectorstore
    finally:
        raggy._vectorstore = None

    assert rebuilt is True
    assert calls["close"] == 1
    assert store is new_store


def test_refresh_db_skips_rebuild_when_fresh(monkeypatch):
    fake_cfg = {
        "sources": ["./docs"],
        "persist_directory": "./persist",
        "chunk_size": 100,
        "chunk_overlap": 10,
        "embedding_model": "embed-x",
    }
    calls = {"close": 0, "init": 0}

    monkeypatch.setattr(raggy, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(raggy, "build_index_config", lambda **kwargs: {"fake": True})
    monkeypatch.setattr(raggy, "db_needs_rebuild", lambda *_: False)
    monkeypatch.setattr(
        raggy,
        "close_vectorstore",
        lambda store: calls.__setitem__("close", calls["close"] + 1),
    )
    monkeypatch.setattr(raggy, "_init_db", lambda: calls.__setitem__("init", 1))
    raggy._vectorstore = object()

    try:
        rebuilt = raggy.refresh_db()
    finally:
        raggy._vectorstore = None

    assert rebuilt is False
    assert calls["close"] == 0
    assert calls["init"] == 0


def test_run_pipeline_stream_yields_chunks_and_captures_docs(monkeypatch):
    fake_cfg = {
        "llm_model": "llm-x",
        "llm_provider": "ollama",
        "system_prompt": "sys",
        "search_type": "mmr",
        "retrieve_k": 2,
        "mmr_fetch_k": 10,
        "temperature": 0.0,
        "relevance_filter": True,
        "rerank_enabled": False,
        "rerank_model": "rerank-x",
        "rerank_k": 2,
    }

    class FakeChain:
        def stream(self, inputs):
            yield "hel"
            yield "lo"

    def fake_build_rag_chain(**kwargs):
        sink = kwargs["doc_sink"]
        sink.append([SimpleNamespace(page_content="doc1")])
        return FakeChain(), SimpleNamespace(invoke=lambda q: [])

    monkeypatch.setattr(raggy, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(raggy, "_get_vectorstore", lambda: object())
    monkeypatch.setattr(raggy, "ensure_ollama_model", lambda model: False)
    monkeypatch.setattr(raggy, "build_rag_chain", fake_build_rag_chain)

    doc_sink = []
    chunks = list(raggy.run_pipeline_stream("hello", doc_sink=doc_sink))

    assert chunks == ["hel", "lo"]
    assert [doc.page_content for doc in doc_sink[-1]] == ["doc1"]


def test_run_pipeline_stream_forwards_chat_history(monkeypatch):
    fake_cfg = {
        "llm_model": "llm-x",
        "llm_provider": "ollama",
        "system_prompt": "sys",
        "search_type": "mmr",
        "retrieve_k": 2,
        "mmr_fetch_k": 10,
        "temperature": 0.0,
        "relevance_filter": True,
        "rerank_enabled": False,
        "rerank_model": "rerank-x",
        "rerank_k": 2,
    }
    captured = {}
    history = [("human", "hello"), ("ai", "hi")]

    class FakeChain:
        def stream(self, inputs):
            captured["input"] = inputs
            yield "ans"

    def fake_build_rag_chain(**kwargs):
        captured["chat_history"] = kwargs["chat_history"]
        return FakeChain(), SimpleNamespace(invoke=lambda q: [])

    monkeypatch.setattr(raggy, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(raggy, "_get_vectorstore", lambda: object())
    monkeypatch.setattr(raggy, "ensure_ollama_model", lambda model: False)
    monkeypatch.setattr(raggy, "build_rag_chain", fake_build_rag_chain)

    list(raggy.run_pipeline_stream("follow-up", chat_history=history))

    assert captured["chat_history"] is history
    assert captured["input"] == {"question": "follow-up", "chat_history": history}


def test_run_pipeline_propagates_unexpected_error(monkeypatch):
    fake_cfg = {
        "llm_model": "llm-x",
        "llm_provider": "ollama",
        "system_prompt": "sys",
        "search_type": "mmr",
        "retrieve_k": 2,
        "mmr_fetch_k": 10,
        "temperature": 0.0,
        "relevance_filter": True,
        "rerank_enabled": False,
        "rerank_model": "rerank-x",
        "rerank_k": 2,
    }

    monkeypatch.setattr(raggy, "load_config", lambda: fake_cfg)
    monkeypatch.setattr(raggy, "_get_vectorstore", lambda: object())
    monkeypatch.setattr(raggy, "ensure_ollama_model", lambda model: False)

    def boom(**_):
        raise RuntimeError("broken")

    monkeypatch.setattr(raggy, "build_rag_chain", boom)

    with pytest.raises(RuntimeError, match="broken"):
        raggy.run_pipeline("hello")
