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
        "hybrid_search": True,
        "hybrid_alpha": 0.5,
        "rerank_enabled": True,
        "rerank_model": "rerank-x",
        "rerank_k": 4,
        "rerank_threshold": 0.3,
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
        "rerank_enabled: false\n"
        "rerank_model: rerank-x\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    cfg = raggy.load_config(str(config_file))

    assert cfg["llm_provider"] == "ollama"


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
        "rerank_enabled: true\n"
        "rerank_model: rerank-x\n"
        "rerank_k: 6\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rerank_k"):
        raggy.load_config(str(config_file))


def test_load_config_hybrid_search_defaults_true_and_alpha_half(tmp_path):
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
        "rerank_enabled: false\n"
        "rerank_model: rerank-x\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    cfg = raggy.load_config(str(config_file))

    assert cfg["hybrid_search"] is True
    assert cfg["hybrid_alpha"] == 0.5


def test_load_config_rejects_hybrid_alpha_out_of_range(tmp_path):
    for bad in ("-0.1", "1.1"):
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
            f"hybrid_alpha: {bad}\n"
            "rerank_enabled: false\n"
            "rerank_model: rerank-x\n"
            "system_prompt: use this\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="hybrid_alpha"):
            raggy.load_config(str(config_file))


def test_load_config_reads_rerank_threshold(tmp_path):
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
        "rerank_enabled: true\n"
        "rerank_model: rerank-x\n"
        "rerank_k: 5\n"
        "rerank_threshold: 0.3\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    cfg = raggy.load_config(str(config_file))

    assert cfg["rerank_threshold"] == 0.3


def test_load_config_rejects_rerank_threshold_out_of_range(tmp_path):
    for bad in ("-0.1", "1.1"):
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
            "rerank_enabled: true\n"
            "rerank_model: rerank-x\n"
            "rerank_k: 5\n"
            f"rerank_threshold: {bad}\n"
            "system_prompt: use this\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="rerank_threshold"):
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
        "rerank_enabled: false\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="llm_provider"):
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
        "rerank_enabled: false\n"
        "system_prompt: use this\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="temperature"):
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

    monkeypatch.setattr(raggy, "load_config", lambda *a, **k: fake_cfg)
    monkeypatch.setattr(
        raggy, "ensure_ollama_model", lambda model, progress=None: False
    )

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
        "progress": None,
    }


def test_run_pipeline_returns_response_and_retrieved_docs(monkeypatch):
    fake_cfg = {
        "llm_model": "llm-x",
        "llm_provider": "ollama",
        "system_prompt": "sys",
        "retrieve_k": 2,
        "temperature": 0.0,
        "persist_directory": "./persist",
        "hybrid_search": True,
        "hybrid_alpha": 0.5,
        "rerank_enabled": False,
        "rerank_model": "rerank-x",
        "rerank_k": 2,
        "rerank_threshold": 0.0,
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

    monkeypatch.setattr(raggy, "load_config", lambda *a, **k: fake_cfg)
    monkeypatch.setattr(raggy, "_get_vectorstore", lambda *a, **k: object())
    monkeypatch.setattr(
        raggy, "ensure_ollama_model", lambda model, progress=None: False
    )
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
        "retrieve_k": 2,
        "temperature": 0.0,
        "persist_directory": "./persist",
        "hybrid_search": True,
        "hybrid_alpha": 0.5,
        "rerank_enabled": False,
        "rerank_model": "rerank-x",
        "rerank_k": 2,
        "rerank_threshold": 0.0,
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

    monkeypatch.setattr(raggy, "load_config", lambda *a, **k: fake_cfg)
    monkeypatch.setattr(raggy, "_get_vectorstore", lambda *a, **k: object())
    monkeypatch.setattr(
        raggy, "ensure_ollama_model", lambda model, progress=None: False
    )
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

    monkeypatch.setattr(raggy, "load_config", lambda *a, **k: fake_cfg)
    monkeypatch.setattr(raggy, "build_index_config", lambda **kwargs: {"fake": True})
    monkeypatch.setattr(raggy, "db_needs_rebuild", lambda *_: True)
    monkeypatch.setattr(
        raggy,
        "close_vectorstore",
        lambda store: calls.__setitem__("close", calls["close"] + 1),
    )
    monkeypatch.setattr(raggy, "_init_db", lambda *a, **k: new_store)
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

    monkeypatch.setattr(raggy, "load_config", lambda *a, **k: fake_cfg)
    monkeypatch.setattr(raggy, "build_index_config", lambda **kwargs: {"fake": True})
    monkeypatch.setattr(raggy, "db_needs_rebuild", lambda *_: False)
    monkeypatch.setattr(
        raggy,
        "close_vectorstore",
        lambda store: calls.__setitem__("close", calls["close"] + 1),
    )
    monkeypatch.setattr(raggy, "_init_db", lambda *a, **k: calls.__setitem__("init", 1))
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
        "retrieve_k": 2,
        "temperature": 0.0,
        "persist_directory": "./persist",
        "hybrid_search": True,
        "hybrid_alpha": 0.5,
        "rerank_enabled": False,
        "rerank_model": "rerank-x",
        "rerank_k": 2,
        "rerank_threshold": 0.0,
    }

    class FakeChain:
        def stream(self, inputs):
            yield "hel"
            yield "lo"

    def fake_build_rag_chain(**kwargs):
        sink = kwargs["doc_sink"]
        sink.append([SimpleNamespace(page_content="doc1")])
        return FakeChain(), SimpleNamespace(invoke=lambda q: [])

    monkeypatch.setattr(raggy, "load_config", lambda *a, **k: fake_cfg)
    monkeypatch.setattr(raggy, "_get_vectorstore", lambda *a, **k: object())
    monkeypatch.setattr(
        raggy, "ensure_ollama_model", lambda model, progress=None: False
    )
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
        "retrieve_k": 2,
        "temperature": 0.0,
        "persist_directory": "./persist",
        "hybrid_search": True,
        "hybrid_alpha": 0.5,
        "rerank_enabled": False,
        "rerank_model": "rerank-x",
        "rerank_k": 2,
        "rerank_threshold": 0.0,
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

    monkeypatch.setattr(raggy, "load_config", lambda *a, **k: fake_cfg)
    monkeypatch.setattr(raggy, "_get_vectorstore", lambda *a, **k: object())
    monkeypatch.setattr(
        raggy, "ensure_ollama_model", lambda model, progress=None: False
    )
    monkeypatch.setattr(raggy, "build_rag_chain", fake_build_rag_chain)

    list(raggy.run_pipeline_stream("follow-up", chat_history=history))

    assert captured["chat_history"] is history
    assert captured["input"] == {"question": "follow-up", "chat_history": history}


def test_run_pipeline_propagates_unexpected_error(monkeypatch):
    fake_cfg = {
        "llm_model": "llm-x",
        "llm_provider": "ollama",
        "system_prompt": "sys",
        "retrieve_k": 2,
        "temperature": 0.0,
        "persist_directory": "./persist",
        "hybrid_search": True,
        "hybrid_alpha": 0.5,
        "rerank_enabled": False,
        "rerank_model": "rerank-x",
        "rerank_k": 2,
        "rerank_threshold": 0.0,
    }

    monkeypatch.setattr(raggy, "load_config", lambda *a, **k: fake_cfg)
    monkeypatch.setattr(raggy, "_get_vectorstore", lambda *a, **k: object())
    monkeypatch.setattr(
        raggy, "ensure_ollama_model", lambda model, progress=None: False
    )

    def boom(**_):
        raise RuntimeError("broken")

    monkeypatch.setattr(raggy, "build_rag_chain", boom)

    with pytest.raises(RuntimeError, match="broken"):
        raggy.run_pipeline("hello")


def test_ensure_models_pulls_embedding_and_local_llm(monkeypatch):
    pulled = []
    monkeypatch.setattr(
        raggy,
        "ensure_ollama_model",
        lambda model, progress=None: pulled.append((model, progress)),
    )
    monkeypatch.setattr(
        raggy, "ensure_reranker_model", lambda model, progress=None: None
    )

    sentinel = object()
    raggy.ensure_models(
        {
            "embedding_model": "embed-x",
            "llm_provider": "ollama",
            "llm_model": "llm-x",
            "rerank_enabled": False,
            "rerank_model": "rerank-x",
        },
        progress=sentinel,
    )

    assert pulled == [("embed-x", sentinel), ("llm-x", sentinel)]


def test_ensure_models_skips_llm_for_remote_providers(monkeypatch):
    pulled = []
    monkeypatch.setattr(
        raggy,
        "ensure_ollama_model",
        lambda model, progress=None: pulled.append(model),
    )
    monkeypatch.setattr(
        raggy, "ensure_reranker_model", lambda model, progress=None: None
    )

    # Embeddings are always local, so the embedding model is still pulled.
    raggy.ensure_models(
        {
            "embedding_model": "embed-x",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "rerank_enabled": False,
            "rerank_model": "rerank-x",
        }
    )

    assert pulled == ["embed-x"]


def test_ensure_models_downloads_reranker_when_enabled(monkeypatch):
    downloaded = []
    monkeypatch.setattr(raggy, "ensure_ollama_model", lambda model, progress=None: None)
    monkeypatch.setattr(
        raggy,
        "ensure_reranker_model",
        lambda model, progress=None: downloaded.append((model, progress)),
    )

    sentinel = object()
    raggy.ensure_models(
        {
            "embedding_model": "embed-x",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "rerank_enabled": True,
            "rerank_model": "rerank-x",
        },
        progress=sentinel,
    )

    assert downloaded == [("rerank-x", sentinel)]


def test_ensure_models_skips_reranker_when_disabled(monkeypatch):
    downloaded = []
    monkeypatch.setattr(raggy, "ensure_ollama_model", lambda model, progress=None: None)
    monkeypatch.setattr(
        raggy,
        "ensure_reranker_model",
        lambda model, progress=None: downloaded.append(model),
    )

    raggy.ensure_models(
        {
            "embedding_model": "embed-x",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
            "rerank_enabled": False,
            "rerank_model": "rerank-x",
        }
    )

    assert downloaded == []


def test_refresh_db_announces_staleness_before_reindexing(monkeypatch):
    fake_cfg = {
        "sources": ["./docs"],
        "persist_directory": "./persist",
        "chunk_size": 100,
        "chunk_overlap": 10,
        "embedding_model": "embed-x",
    }
    events = []

    monkeypatch.setattr(raggy, "load_config", lambda *a, **k: fake_cfg)
    monkeypatch.setattr(raggy, "build_index_config", lambda **kwargs: {"fake": True})
    monkeypatch.setattr(raggy, "db_needs_rebuild", lambda *_: True)
    monkeypatch.setattr(raggy, "close_vectorstore", lambda store: None)
    monkeypatch.setattr(raggy, "_init_db", lambda *a, **k: events.append("reindexed"))

    try:
        raggy.refresh_db(on_stale=lambda: events.append("announced"))
    finally:
        raggy._vectorstore = None

    # The announcement is only useful ahead of the work it describes.
    assert events == ["announced", "reindexed"]


def test_refresh_db_stays_quiet_when_db_is_current(monkeypatch):
    fake_cfg = {
        "sources": ["./docs"],
        "persist_directory": "./persist",
        "chunk_size": 100,
        "chunk_overlap": 10,
        "embedding_model": "embed-x",
    }
    events = []

    monkeypatch.setattr(raggy, "load_config", lambda *a, **k: fake_cfg)
    monkeypatch.setattr(raggy, "build_index_config", lambda **kwargs: {"fake": True})
    monkeypatch.setattr(raggy, "db_needs_rebuild", lambda *_: False)

    assert raggy.refresh_db(on_stale=lambda: events.append("announced")) is False
    assert events == []


def test_run_pipeline_stream_runs_against_the_given_config(monkeypatch):
    """A custom config path reaches both the config read and the vectorstore."""
    fake_cfg = {
        "llm_model": "llm-x",
        "llm_provider": "openai",
        "system_prompt": "sys",
        "retrieve_k": 2,
        "temperature": 0.0,
        "persist_directory": "./persist",
        "hybrid_search": True,
        "hybrid_alpha": 0.5,
        "rerank_enabled": False,
        "rerank_model": "rerank-x",
        "rerank_k": 2,
        "rerank_threshold": 0.0,
    }
    seen = {}

    class FakeChain:
        def stream(self, inputs):
            yield "hi"

    def fake_load_config(path="config.yaml"):
        seen["load"] = path
        return fake_cfg

    def fake_get_vectorstore(path="config.yaml"):
        seen["store"] = path
        return object()

    monkeypatch.setattr(raggy, "load_config", fake_load_config)
    monkeypatch.setattr(raggy, "_get_vectorstore", fake_get_vectorstore)
    monkeypatch.setattr(
        raggy, "build_rag_chain", lambda **kwargs: (FakeChain(), object())
    )

    list(raggy.run_pipeline_stream("hello", config_path="/tmp/other.yaml"))

    assert seen == {"load": "/tmp/other.yaml", "store": "/tmp/other.yaml"}


def test_refresh_db_reindexes_from_the_given_config(monkeypatch):
    """The rebuild re-reads the caller's config, not the working-directory one."""
    fake_cfg = {
        "sources": ["./docs"],
        "persist_directory": "./persist",
        "chunk_size": 100,
        "chunk_overlap": 10,
        "embedding_model": "embed-x",
    }
    seen = {}

    monkeypatch.setattr(raggy, "build_index_config", lambda **kwargs: {"fake": True})
    monkeypatch.setattr(raggy, "db_needs_rebuild", lambda *_: True)
    monkeypatch.setattr(
        raggy,
        "_init_db",
        lambda progress=None, config_path=None: seen.setdefault("init", config_path),
    )

    try:
        raggy.refresh_db(fake_cfg, config_path="/tmp/other.yaml")
    finally:
        raggy._vectorstore = None

    assert seen["init"] == "/tmp/other.yaml"


def test_init_db_reads_the_given_config(monkeypatch):
    fake_cfg = {
        "persist_directory": "./persist",
        "embedding_model": "embed-x",
        "sources": ["./docs"],
        "chunk_size": 100,
        "chunk_overlap": 10,
        "batch_size": 50,
    }
    seen = {}

    def fake_load_config(path="config.yaml"):
        seen["load"] = path
        return fake_cfg

    monkeypatch.setattr(raggy, "load_config", fake_load_config)
    monkeypatch.setattr(
        raggy, "ensure_ollama_model", lambda model, progress=None: False
    )
    monkeypatch.setattr(raggy, "initialize_db", lambda **kwargs: object())

    raggy._init_db(config_path="/tmp/other.yaml")

    assert seen["load"] == "/tmp/other.yaml"
