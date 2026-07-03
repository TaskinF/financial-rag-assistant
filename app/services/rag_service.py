from __future__ import annotations

from app.api.schemas import AskRequest, AskResponse
from app.config import settings
from app.llm.llm_client import FakeLLMClient, LLMClient
from app.llm.ollama_client import OllamaLLMClient
from app.processing.ingestion import build_chunks_from_pdf
from app.rag.answer_generator import generate_answer
from app.rag.retriever import VectorStoreRetriever
from app.vectorstore.embedding_cache import (
    build_embedding_cache_key,
    load_embedding_cache,
    save_embedding_cache,
)
from app.vectorstore.embeddings import BGEFlagEmbeddingModel
from app.vectorstore.in_memory_store import InMemoryVectorStore


class RAGService:
    """
    Service layer that orchestrates the local RAG pipeline for API requests.
    """

    def __init__(self, cache_dir: str = "artifacts/embedding_cache") -> None:
        """
        Initialize the RAG service with lazy-loaded shared dependencies.

        Args:
            cache_dir: Directory used to store document embedding cache files.
        """
        self.cache_dir = cache_dir
        self._embedding_model: BGEFlagEmbeddingModel | None = None

    def answer_question(self, request: AskRequest) -> AskResponse:
        """
        Build chunks, retrieve relevant context, and generate an answer.

        Args:
            request: Validated API request payload.

        Returns:
            Structured API response containing the answer and sources.
        """
        chunks = build_chunks_from_pdf(
            file_path=request.pdf_path,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        chunks = self._filter_chunks_by_page(
            chunks=chunks,
            start_page=request.start_page,
            end_page=request.end_page,
        )

        embedding_model = self._get_embedding_model()
        vector_store = InMemoryVectorStore(embedding_model=embedding_model)

        cache_key = build_embedding_cache_key(
            chunks=chunks,
            model_name=embedding_model.model_name,
        )
        cached_documents = load_embedding_cache(self.cache_dir, cache_key)

        if cached_documents is not None:
            vector_store.add_embedded_documents(cached_documents)
        else:
            vector_store.add_documents(chunks)
            save_embedding_cache(
                cache_dir=self.cache_dir,
                cache_key=cache_key,
                documents=vector_store.get_documents(),
            )

        retriever = VectorStoreRetriever(vector_store)
        results = retriever.retrieve(request.question, top_k=request.top_k)
        answer_chunks = results[: request.answer_top_k]

        llm_client, response_model_name = self._build_llm_client(request)
        answer_payload = generate_answer(
            question=request.question,
            retrieved_chunks=answer_chunks,
            llm_client=llm_client,
            max_context_chars=request.max_context_chars,
        )

        return AskResponse(
            question=answer_payload["question"],
            answer=answer_payload["answer"],
            sources=answer_payload["sources"],
            retrieved_count=len(results),
            answer_context_count=len(answer_chunks),
            llm_provider=request.llm_provider,
            llm_model=response_model_name,
        )

    def _get_embedding_model(self) -> BGEFlagEmbeddingModel:
        """
        Lazily create and reuse the shared embedding model instance.
        """
        if self._embedding_model is None:
            # Keep the API path conservative for CPU-first local setups.
            self._embedding_model = BGEFlagEmbeddingModel(use_fp16=False)

        return self._embedding_model

    def _build_llm_client(self, request: AskRequest) -> tuple[LLMClient, str]:
        """
        Build the requested LLM client for answer generation.

        Args:
            request: Validated API request payload.

        Returns:
            Tuple of client instance and response model label.
        """
        if request.llm_provider == "fake":
            return FakeLLMClient(response="Fake answer preview"), "fake"

        return (
            OllamaLLMClient(
                base_url=settings.ollama_base_url,
                model=request.llm_model,
            ),
            request.llm_model,
        )

    def _filter_chunks_by_page(
        self,
        chunks: list[dict],
        start_page: int | None,
        end_page: int | None,
    ) -> list[dict]:
        """
        Filter chunk list by optional page range while preserving order.

        Args:
            chunks: Chunk dictionaries containing page metadata.
            start_page: Inclusive lower page bound.
            end_page: Inclusive upper page bound.

        Returns:
            Filtered chunk list.
        """
        if start_page is None and end_page is None:
            return chunks

        filtered_chunks: list[dict] = []

        for chunk in chunks:
            page_number = chunk.get("page_number")

            if page_number is None:
                continue

            if start_page is not None and page_number < start_page:
                continue

            if end_page is not None and page_number > end_page:
                continue

            filtered_chunks.append(chunk)

        return filtered_chunks
