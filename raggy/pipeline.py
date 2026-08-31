from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from .llm_factory import get_llm
from .llm_filter import filter_docs_by_relevance
from .reranker import DEFAULT_RERANKER_MODEL, get_cross_encoder

_ROLE_LABELS = {"human": "User", "ai": "Assistant", "system": "System"}

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


def get_retriever(
    vectorstore: Chroma,
    search_type: str,
    k: int,
    fetch_k: int,
    rerank_enabled: bool = False,
    rerank_model: str = DEFAULT_RERANKER_MODEL,
    rerank_k: int | None = None,
):
    """Configures and returns the retriever.

    The pipeline is two-stage:

      1. First stage (always): vector retrieval over the whole corpus. With
         ``search_type: "mmr"`` this is an MMR pass that returns ``k`` chunks
         while considering ``fetch_k`` candidates. ``k`` and ``fetch_k``
         always apply to this stage, whether or not re-ranking is enabled.
      2. Second stage (optional): a cross-encoder scores each
         (query, chunk) pair and keeps the top ``rerank_k`` chunks.
         ``rerank_k`` defaults to ``k`` and must not exceed ``k``.
    """
    search_kwargs: dict = {"k": k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k
    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )

    if rerank_enabled:
        final_k = rerank_k if rerank_k is not None else k
        compressor = CrossEncoderReranker(
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
    search_type: str,
    k: int,
    fetch_k: int,
    temperature: float,
    relevance_filter: bool = False,
    rerank_enabled: bool = False,
    rerank_model: str = DEFAULT_RERANKER_MODEL,
    rerank_k: int | None = None,
    doc_sink: list | None = None,
    chat_history: list | None = None,
) -> tuple[Runnable, Runnable]:
    """
    Builds the RAG Chain using LangChain Expression Language (LCEL).
    Returns a tuple containing (rag_chain, retriever).

    When ``relevance_filter`` is true, the LLM grades each chunk as
    relevant/irrelevant in a single pass and only the relevant ones reach
    the prompt; it is off by default. The surviving Documents are captured in a single
    pass and appended to ``doc_sink`` (if provided), so callers can inspect
    exactly what the LLM saw without running the retriever a second time.

    When ``chat_history`` is provided (a list of ``("human"|"ai", content)``
    tuples), the chain is memory-aware: retrieval runs against a standalone
    question condensed from the follow-up plus the history, and the final
    prompt includes the full history. The chain must then be invoked with a
    dict of ``{"question": str, "chat_history": [...]}``.
    """
    retriever = get_retriever(
        vectorstore,
        search_type=search_type,
        k=k,
        fetch_k=fetch_k,
        rerank_enabled=rerank_enabled,
        rerank_model=rerank_model,
        rerank_k=rerank_k,
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
        if relevance_filter:
            docs = filter_docs_by_relevance(search_query, docs, llm)
        return format_and_capture(docs)

    context_step: Runnable = RunnableLambda(select_context)

    rag_chain = (
        RunnablePassthrough.assign(context=context_step)
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever
