from types import SimpleNamespace

from raggy import pipeline


def test_get_retriever_uses_vectorstore_configuration():
    captured = {}

    class FakeVectorstore:
        def as_retriever(self, **kwargs):
            captured.update(kwargs)
            return "retriever"

    result = pipeline.get_retriever(
        FakeVectorstore(), search_type="similarity", k=8, fetch_k=10
    )

    assert result == "retriever"
    assert captured == {"search_type": "similarity", "search_kwargs": {"k": 8}}


def test_get_retriever_wraps_cross_encoder_when_rerank_enabled(monkeypatch):
    captured = {}

    class FakeVectorstore:
        def as_retriever(self, **kwargs):
            captured["base"] = kwargs
            return "base-retriever"

    def fake_compressor_cls(**kwargs):
        captured["compressor"] = kwargs
        return "compressor"

    def fake_compression_retriever(**kwargs):
        captured["compression"] = kwargs
        return "compressed-retriever"

    monkeypatch.setattr(pipeline, "CrossEncoderReranker", fake_compressor_cls)
    monkeypatch.setattr(
        pipeline, "ContextualCompressionRetriever", fake_compression_retriever
    )
    monkeypatch.setattr(pipeline, "get_cross_encoder", lambda model: f"encoder:{model}")

    result = pipeline.get_retriever(
        FakeVectorstore(),
        search_type="similarity",
        k=5,
        fetch_k=25,
        rerank_enabled=True,
        rerank_model="reranker-model",
    )

    assert result == "compressed-retriever"
    assert captured["base"] == {"search_type": "similarity", "search_kwargs": {"k": 5}}
    assert captured["compressor"] == {
        "model": "encoder:reranker-model",
        "top_n": 5,
    }
    assert captured["compression"] == {
        "base_compressor": "compressor",
        "base_retriever": "base-retriever",
    }


def test_get_retriever_mmr_always_uses_k_and_fetch_k_with_rerank_k(monkeypatch):
    captured = {}

    class FakeVectorstore:
        def as_retriever(self, **kwargs):
            captured["base"] = kwargs
            return "base-retriever"

    def fake_compressor_cls(**kwargs):
        captured["compressor"] = kwargs
        return "compressor"

    def fake_compression_retriever(**kwargs):
        captured["compression"] = kwargs
        return "compressed-retriever"

    monkeypatch.setattr(pipeline, "CrossEncoderReranker", fake_compressor_cls)
    monkeypatch.setattr(
        pipeline, "ContextualCompressionRetriever", fake_compression_retriever
    )
    monkeypatch.setattr(pipeline, "get_cross_encoder", lambda model: f"encoder:{model}")

    result = pipeline.get_retriever(
        FakeVectorstore(),
        search_type="mmr",
        k=5,
        fetch_k=25,
        rerank_enabled=True,
        rerank_model="reranker-model",
        rerank_k=3,
    )

    assert result == "compressed-retriever"
    assert captured["base"] == {
        "search_type": "mmr",
        "search_kwargs": {"k": 5, "fetch_k": 25},
    }
    assert captured["compressor"] == {
        "model": "encoder:reranker-model",
        "top_n": 3,
    }


def test_format_docs_joins_page_content():
    docs = [SimpleNamespace(page_content="A"), SimpleNamespace(page_content="B")]
    assert pipeline.format_docs(docs) == "A\n\nB"


def test_get_prompt_template_calls_from_messages(monkeypatch):
    captured = {}

    class FakePromptTemplate:
        @staticmethod
        def from_messages(messages):
            captured["messages"] = messages
            return "prompt-template"

    monkeypatch.setattr(pipeline, "ChatPromptTemplate", FakePromptTemplate)

    result = pipeline.get_prompt_template("use context")

    assert result == "prompt-template"
    assert captured["messages"] == [("system", "use context"), ("human", "{question}")]
