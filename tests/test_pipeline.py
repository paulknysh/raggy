from types import SimpleNamespace

from raggy import pipeline


def _message_contents(messages):
    """Flatten (role, content) tuples / BaseMessage objects into their content."""
    contents = []
    for m in messages:
        if isinstance(m, list):
            contents.extend(_message_contents(m))
        elif isinstance(m, tuple):
            contents.append(m[1])
        elif isinstance(m, str):
            contents.append(m)
        else:
            contents.append(m.content)
    return contents


class FakeLLM:
    """Chain-compatible fake chat model: callable (for ``| llm``) and ``.invoke``."""

    def __init__(self, capture, output="final answer"):
        self.capture = capture
        self.output = output

    def __call__(self, prompt_input):
        self.capture["prompt_messages"] = self._normalize(prompt_input)
        return self.output

    def invoke(self, messages):
        self.capture["prompt_messages"] = self._normalize(messages)
        return self.output

    @staticmethod
    def _normalize(prompt_input):
        if hasattr(prompt_input, "to_messages"):
            return prompt_input.to_messages()
        if isinstance(prompt_input, list):
            messages = []
            for m in prompt_input:
                if not isinstance(m, (str, tuple)):
                    messages.extend(FakeLLM._normalize(m))
                else:
                    messages.append(m)
            return messages
        return prompt_input


def test_get_retriever_uses_vectorstore_configuration():
    captured = {}

    class FakeVectorstore:
        def as_retriever(self, **kwargs):
            captured.update(kwargs)
            return "retriever"

    result = pipeline.get_retriever(
        FakeVectorstore(),
        search_type="similarity",
        retrieve_k=8,
        mmr_fetch_k=10,
        rerank_model="test-reranker",
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
        retrieve_k=5,
        mmr_fetch_k=25,
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
        retrieve_k=5,
        mmr_fetch_k=25,
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


def test_get_prompt_template_adds_history_placeholder(monkeypatch):
    captured = {}

    class FakePromptTemplate:
        @staticmethod
        def from_messages(messages):
            captured["messages"] = messages
            return "prompt-template"

    monkeypatch.setattr(pipeline, "ChatPromptTemplate", FakePromptTemplate)

    pipeline.get_prompt_template("use context", with_history=True)

    assert captured["messages"][0] == ("system", "use context")
    assert captured["messages"][1].variable_name == "chat_history"
    assert captured["messages"][2] == ("human", "{question}")


def test_condense_question_passthrough_without_history():
    assert pipeline.condense_question([], "plain question", "llm-not-called") == (
        "plain question"
    )


def test_condense_question_rewrites_with_history(monkeypatch):
    captured = {}

    def fake_invoke(messages):
        captured["messages"] = messages
        return SimpleNamespace(content="standalone question")

    class FakeParser:
        def invoke(self, message):
            return message.content

    monkeypatch.setattr(pipeline, "StrOutputParser", lambda: FakeParser())

    history = [("human", "What is the pricing?"), ("ai", "It starts at $10.")]
    fake_llm = SimpleNamespace(invoke=fake_invoke)
    result = pipeline.condense_question(
        history, "what about the annual plan?", fake_llm
    )

    assert result == "standalone question"
    rendered = captured["messages"][1].content
    assert "User: What is the pricing?" in rendered
    assert "Assistant: It starts at $10." in rendered


def test_build_rag_chain_threads_history(monkeypatch):
    captured = {}
    fake_llm = FakeLLM(captured)
    monkeypatch.setattr(pipeline, "get_llm", lambda *a, **k: fake_llm)
    monkeypatch.setattr(
        pipeline,
        "get_retriever",
        lambda *a, **k: SimpleNamespace(
            invoke=lambda q: [SimpleNamespace(page_content="doc")]
        ),
    )
    monkeypatch.setattr(pipeline, "filter_docs_by_relevance", lambda q, docs, llm: docs)

    def fake_condense(history, question, llm):
        captured["history"] = history
        captured["question"] = question
        return "standalone"

    monkeypatch.setattr(pipeline, "condense_question", fake_condense)

    history = [("human", "prev question"), ("ai", "prev answer")]
    chain, _ = pipeline.build_rag_chain(
        vectorstore=object(),
        llm_model="m",
        llm_provider="ollama",
        system_prompt="sys",
        search_type="similarity",
        retrieve_k=5,
        mmr_fetch_k=25,
        temperature=0.0,
        rerank_model="test-reranker",
        chat_history=history,
    )

    out = chain.invoke({"question": "follow-up", "chat_history": history})

    assert out == "final answer"
    assert captured["history"] == history
    assert captured["question"] == "follow-up"
    assert _message_contents(captured["prompt_messages"]) == [
        "sys",
        "prev question",
        "prev answer",
        "follow-up",
    ]


def test_build_rag_chain_without_history_skips_condense(monkeypatch):
    captured = {}
    fake_llm = FakeLLM(captured)
    monkeypatch.setattr(pipeline, "get_llm", lambda *a, **k: fake_llm)
    monkeypatch.setattr(
        pipeline,
        "get_retriever",
        lambda *a, **k: SimpleNamespace(
            invoke=lambda q: [SimpleNamespace(page_content="doc")]
        ),
    )
    monkeypatch.setattr(pipeline, "filter_docs_by_relevance", lambda q, docs, llm: docs)

    def fake_condense(*args):
        captured["condense_called"] = True
        return "condensed"

    monkeypatch.setattr(pipeline, "condense_question", fake_condense)

    chain, _ = pipeline.build_rag_chain(
        vectorstore=object(),
        llm_model="m",
        llm_provider="ollama",
        system_prompt="sys",
        search_type="similarity",
        retrieve_k=5,
        mmr_fetch_k=25,
        temperature=0.0,
        rerank_model="test-reranker",
    )

    out = chain.invoke({"question": "plain", "chat_history": []})

    assert out == "final answer"
    assert "condense_called" not in captured
    assert _message_contents(captured["prompt_messages"]) == ["sys", "plain"]


def test_build_rag_chain_skips_relevance_filter_when_disabled(monkeypatch):
    captured = {"filter_called": False}
    fake_llm = FakeLLM(captured)
    monkeypatch.setattr(pipeline, "get_llm", lambda *a, **k: fake_llm)
    monkeypatch.setattr(
        pipeline,
        "get_retriever",
        lambda *a, **k: SimpleNamespace(
            invoke=lambda q: [SimpleNamespace(page_content="doc")]
        ),
    )

    def fake_filter(query, docs, llm):
        captured["filter_called"] = True
        return docs

    monkeypatch.setattr(pipeline, "filter_docs_by_relevance", fake_filter)

    chain, _ = pipeline.build_rag_chain(
        vectorstore=object(),
        llm_model="m",
        llm_provider="ollama",
        system_prompt="sys",
        search_type="similarity",
        retrieve_k=5,
        mmr_fetch_k=25,
        temperature=0.0,
        rerank_model="test-reranker",
        relevance_filter=False,
    )

    out = chain.invoke({"question": "plain", "chat_history": []})

    assert out == "final answer"
    assert captured["filter_called"] is False


def test_build_rag_chain_applies_relevance_filter_when_enabled(monkeypatch):
    captured = {"filter_called": False}
    fake_llm = FakeLLM(captured)
    monkeypatch.setattr(pipeline, "get_llm", lambda *a, **k: fake_llm)
    monkeypatch.setattr(
        pipeline,
        "get_retriever",
        lambda *a, **k: SimpleNamespace(
            invoke=lambda q: [SimpleNamespace(page_content="doc")]
        ),
    )

    def fake_filter(query, docs, llm):
        captured["filter_called"] = True
        return docs

    monkeypatch.setattr(pipeline, "filter_docs_by_relevance", fake_filter)

    chain, _ = pipeline.build_rag_chain(
        vectorstore=object(),
        llm_model="m",
        llm_provider="ollama",
        system_prompt="sys",
        search_type="similarity",
        retrieve_k=5,
        mmr_fetch_k=25,
        temperature=0.0,
        rerank_model="test-reranker",
        relevance_filter=True,
    )

    chain.invoke({"question": "plain", "chat_history": []})

    assert captured["filter_called"] is True
