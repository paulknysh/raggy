from collections.abc import Sequence

from langchain_chroma import Chroma
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pydantic import ConfigDict

from .bm25 import get_bm25_retriever
from .llm_factory import get_llm
from .reranker import get_cross_encoder

_ROLE_LABELS = {"human": "User", "ai": "Assistant", "system": "System"}

SCORE_KEY = "relevance_score"

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Given the conversation and the follow-up question, rephrase the "
                "follow-up question into a standalone question that captures all "
                "the context needed to answer it. Return only the standalone "
                "question."
            ),
        ),
        (
            "human",
            "Conversation:\n{chat_history}\n\nFollow-up question: {question}",
        ),
    ]
)


def _format_history(chat_history: list[tuple[str, str]]) -> str:
    """Render message tuples as readable "Role: content" lines."""
    return "\n".join(
        f"{_ROLE_LABELS.get(role, role)}: {content}" for role, content in chat_history
    )


def condense_question(chat_history: list[tuple[str, str]], question: str, llm) -> str:
    """Rewrite a follow-up question into a standalone question.

    A follow-up ("what about its price?") only retrieves well once it carries
    the context from the earlier exchange, so the chat LLM rephrases it into a
    standalone question before retrieval. With an empty conversation the
    question is returned unchanged and no LLM call is made; if the model
    returns nothing, the original question is used as a fallback.
    """
    if not chat_history:
        return question
    messages = _CONDENSE_PROMPT.format_messages(
        chat_history=_format_history(chat_history),
        question=question,
    )
    condensed = StrOutputParser().invoke(llm.invoke(messages)).strip()
    return condensed or question


class ScoreAnnotatingReranker(BaseDocumentCompressor):
    """Cross-encoder reranker that keeps each chunk's relevance score.

    Behaves like LangChain's ``CrossEncoderReranker`` (score, sort descending,
    take the top ``top_n``) but additionally writes the score into
    ``doc.metadata[SCORE_KEY]`` for every returned chunk. The stock reranker
    discards the scores once it has selected the top-k, so annotating them is
    what lets :func:`filter_by_score_threshold` drop the low-scoring tail
    before the final generation step.
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


def get_retriever(
    vectorstore: Chroma,
    retrieve_k: int,
    rerank_model: str,
    rerank_enabled: bool = False,
    rerank_k: int | None = None,
    persist_directory: str | None = None,
    hybrid_search: bool = True,
    hybrid_alpha: float = 0.5,
):
    """Configures and returns the retriever.

    The pipeline is two-stage:

      1. First stage (always): similarity retrieval over the whole corpus,
         returning ``retrieve_k`` chunks. When ``hybrid_search`` is set,
         the vector retriever is fused with a lexical ``bm25s`` retriever
         (loaded from the persisted index) via reciprocal rank fusion,
         weighting the dense pass by ``hybrid_alpha``.
       2. Second stage (optional): a cross-encoder scores each
         (query, chunk) pair and keeps the top ``rerank_k`` chunks.
         ``rerank_k`` defaults to ``retrieve_k`` and must not exceed it.
    """
    search_kwargs: dict = {"k": retrieve_k}
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs=search_kwargs
    )

    if hybrid_search:
        bm25_retriever = get_bm25_retriever(persist_directory, k=retrieve_k)
        retriever = EnsembleRetriever(
            retrievers=[retriever, bm25_retriever],
            weights=[hybrid_alpha, 1.0 - hybrid_alpha],
        )

    if rerank_enabled:
        final_k = rerank_k if rerank_k is not None else retrieve_k
        compressor = ScoreAnnotatingReranker(
            model=get_cross_encoder(rerank_model), top_n=final_k
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

    return retriever


def format_docs(retrieved_docs: list[Document]) -> str:
    """Helper function to format retrieved chunks into a single string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def get_prompt_template(
    system_prompt: str, with_history: bool = False
) -> ChatPromptTemplate:
    """Creates and returns the prompt template.

    When ``with_history`` is set, a ``chat_history`` message placeholder is
    inserted between the system prompt and the question so the model also sees
    the prior exchange.
    """
    messages = [("system", system_prompt)]
    if with_history:
        messages.append(MessagesPlaceholder("chat_history"))
    messages.append(("human", "{question}"))
    return ChatPromptTemplate.from_messages(messages)


def build_rag_chain(
    vectorstore: Chroma,
    llm_model: str,
    llm_provider: str,
    system_prompt: str,
    retrieve_k: int,
    temperature: float,
    rerank_model: str,
    rerank_enabled: bool = False,
    rerank_k: int | None = None,
    rerank_threshold: float = 0.0,
    persist_directory: str | None = None,
    hybrid_search: bool = True,
    hybrid_alpha: float = 0.5,
    doc_sink: list | None = None,
    chat_history: list | None = None,
) -> tuple[Runnable, Runnable]:
    """
    Builds the RAG Chain using LangChain Expression Language (LCEL).
    Returns a tuple containing (rag_chain, retriever).

    When ``rerank_threshold`` is above 0 the cross-encoder's relevance score
    (stamped on each chunk by :class:`ScoreAnnotatingReranker`) is used to drop
    any reranked chunk scoring below the threshold before the prompt; it is off
    by default. The
    surviving Documents are captured in a single pass and appended to
    ``doc_sink`` (if provided), so callers can inspect exactly what the LLM
    saw without running the retriever a second time.

    When ``chat_history`` is provided (a list of ``("human"|"ai", content)``
    tuples), the chain is memory-aware: retrieval runs against a standalone
    question condensed from the follow-up plus the history, and the final
    prompt includes the full history. The chain must then be invoked with a
    dict of ``{"question": str, "chat_history": [...]}``.
    """
    retriever = get_retriever(
        vectorstore,
        retrieve_k=retrieve_k,
        rerank_enabled=rerank_enabled,
        rerank_model=rerank_model,
        rerank_k=rerank_k,
        persist_directory=persist_directory,
        hybrid_search=hybrid_search,
        hybrid_alpha=hybrid_alpha,
    )
    llm = get_llm(llm_provider, llm_model, temperature=temperature)
    prompt = get_prompt_template(system_prompt, with_history=chat_history is not None)

    def format_and_capture(docs: list[Document]) -> str:
        if doc_sink is not None:
            doc_sink.append(docs)
        return format_docs(docs)

    def select_context(state: dict) -> str:
        question = state["question"]
        history = state.get("chat_history") or []
        search_query = (
            condense_question(history, question, llm) if history else question
        )
        docs = retriever.invoke(search_query)
        docs = filter_by_score_threshold(docs, rerank_threshold)
        return format_and_capture(docs)

    context_step: Runnable = RunnableLambda(select_context)

    rag_chain = (
        RunnablePassthrough.assign(context=context_step)
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever
