from abc import ABC, abstractmethod


class Retriever(ABC):
    """
    Base interface for retrieval components in the RAG layer.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve the most relevant documents for a query.

        Args:
            query: User query text.
            top_k: Number of top results to return.

        Returns:
            A list of retrieved document dictionaries.
        """
        raise NotImplementedError


class VectorStoreRetriever(Retriever):
    """
    Retriever implementation that delegates search to a vector store.
    """

    def __init__(self, vector_store) -> None:
        """
        Initialize the retriever with a vector store.

        Args:
            vector_store: Any vector store object exposing a similarity_search method.
        """
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve top matching documents from the vector store.

        Args:
            query: User query text.
            top_k: Number of top results to return.

        Returns:
            A list of retrieved document dictionaries.

        Raises:
            ValueError: If query is empty or top_k is not positive.
        """
        if query is None or not query.strip():
            raise ValueError("query must be a non-empty string")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        return self.vector_store.similarity_search(
            query=query,
            top_k=top_k,
        )