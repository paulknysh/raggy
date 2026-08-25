from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from .reranker import DEFAULT_RERANKER_MODEL, get_cross_encoder


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


def get_llm(model_name: str, temperature: float) -> ChatOllama:
    """Initializes and returns the ChatOllama language model."""
    return ChatOllama(model=model_name, temperature=temperature)


def format_docs(retrieved_docs: list[Document]) -> str:
    """Helper function to format retrieved chunks into a single string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def get_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    """Creates and returns the prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )


def build_rag_chain(
    vectorstore: Chroma,
    llm_model: str,
    system_prompt: str,
    search_type: str,
    k: int,
    fetch_k: int,
    temperature: float,
    rerank_enabled: bool = False,
    rerank_model: str = DEFAULT_RERANKER_MODEL,
    rerank_k: int | None = None,
    doc_sink: list | None = None,
) -> tuple[Runnable, Runnable]:
    """
    Builds the RAG Chain using LangChain Expression Language (LCEL).
    Returns a tuple containing (rag_chain, retriever).

    The retrieved Documents are captured in a single pass and appended to
    ``doc_sink`` (if provided), so callers can inspect exactly what the LLM
    saw without running the retriever a second time.
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
    llm = get_llm(llm_model, temperature=temperature)
    prompt = get_prompt_template(system_prompt)

    def format_and_capture(docs: list[Document]) -> str:
        if doc_sink is not None:
            doc_sink.append(docs)
        return format_docs(docs)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_and_capture),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever
