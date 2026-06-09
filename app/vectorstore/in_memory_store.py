import math

from app.vectorstore.embeddings import EmbeddingModel


class InMemoryVectorStore:
    """
    Simple in-memory vector store for chunk embeddings and similarity search.
    """

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        """
        Initialize the in-memory vector store.

        Args:
            embedding_model: Embedding model used to generate query and document embeddings.
        """
        self.embedding_model = embedding_model
        self._documents: list[dict] = []

    def add_documents(self, chunks: list[dict]) -> None:
        """
        Add chunk documents to the store with their embeddings.

        Args:
            chunks: List of metadata-preserving chunk dictionaries.

        Raises:
            ValueError: If any chunk has missing or empty text.
        """
        if not chunks:
            return

        texts: list[str] = []

        for chunk in chunks:
            text = chunk.get("text")

            if text is None or not str(text).strip():
                raise ValueError("each chunk must contain a non-empty text field")

            texts.append(str(text).strip())

        embeddings = self.embedding_model.embed_documents(texts)

        for chunk, embedding, text in zip(chunks, embeddings, texts):
            stored_document = dict(chunk)
            stored_document["text"] = text
            stored_document["embedding"] = embedding
            self._documents.append(stored_document)

    def similarity_search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search for the most similar stored chunks to the given query.

        Args:
            query: Query text.
            top_k: Number of top results to return.

        Returns:
            List of top matching chunk dictionaries with similarity scores.

        Raises:
            ValueError: If query is empty or top_k is not positive.
        """
        if query is None or not query.strip():
            raise ValueError("query must be a non-empty string")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not self._documents:
            return []

        query_embedding = self.embedding_model.embed_text(query.strip())
        scored_results: list[dict] = []

        for document in self._documents:
            score = self._cosine_similarity(query_embedding, document["embedding"])

            result = {
                key: value
                for key, value in document.items()
                if key != "embedding"
            }
            result["score"] = score
            scored_results.append(result)

        scored_results.sort(key=lambda item: item["score"], reverse=True)
        return scored_results[:top_k]

    def _cosine_similarity(self, vector_a: list[float], vector_b: list[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vector_a: First vector.
            vector_b: Second vector.

        Returns:
            Cosine similarity score.

        Raises:
            ValueError: If vector dimensions do not match.
        """
        if len(vector_a) != len(vector_b):
            raise ValueError("vector dimensions must match")

        norm_a = math.sqrt(sum(value * value for value in vector_a))
        norm_b = math.sqrt(sum(value * value for value in vector_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        return dot_product / (norm_a * norm_b)