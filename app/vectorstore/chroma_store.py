from __future__ import annotations

from pathlib import Path

import chromadb


class ChromaVectorStore:
    """
    Persistent ChromaDB-backed vector store for financial document chunks.
    """

    def __init__(
        self,
        embedding_model,
        persist_dir: str = "artifacts/chroma",
        collection_name: str = "financial_documents",
    ) -> None:
        """
        Initialize the persistent Chroma vector store.

        Args:
            embedding_model: Embedding model used for document and query embeddings.
            persist_dir: Directory where Chroma persistence files are stored.
            collection_name: Chroma collection name.

        Raises:
            ValueError: If embedding_model is None.
        """
        if embedding_model is None:
            raise ValueError("embedding_model cannot be None")

        self.embedding_model = embedding_model
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
        )

    def add_documents(self, chunks: list[dict]) -> None:
        """
        Embed and upsert chunk documents into Chroma.

        Args:
            chunks: Metadata-preserving chunk dictionaries.

        Raises:
            ValueError: If any chunk has missing or empty text.
        """
        if not chunks:
            return

        texts: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []

        for index, chunk in enumerate(chunks):
            text = self._validate_text(chunk.get("text"), "each chunk must contain a non-empty text field")
            chunk_id = self._resolve_chunk_id(chunk, index=index)

            texts.append(text)
            ids.append(chunk_id)
            metadatas.append(self._build_metadata(chunk, chunk_id))

        embeddings = self.embedding_model.embed_documents(texts)
        normalized_embeddings = [
            [float(value) for value in embedding]
            for embedding in embeddings
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=normalized_embeddings,
            metadatas=metadatas,
        )

    def add_embedded_documents(self, documents: list[dict]) -> None:
        """
        Upsert pre-embedded documents into Chroma without recomputing embeddings.

        Args:
            documents: Document dictionaries containing text, embedding, and metadata.

        Raises:
            ValueError: If text or embedding fields are missing or invalid.
        """
        if not documents:
            return

        texts: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []
        embeddings: list[list[float]] = []

        for index, document in enumerate(documents):
            text = self._validate_text(
                document.get("text"),
                "each document must contain a non-empty text field",
            )
            embedding = self._validate_embedding(document.get("embedding"))
            chunk_id = self._resolve_chunk_id(document, index=index)

            texts.append(text)
            ids.append(chunk_id)
            metadatas.append(self._build_metadata(document, chunk_id))
            embeddings.append(embedding)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        Search the Chroma collection for the most similar chunks.

        Args:
            query: Query text.
            top_k: Number of top results to return.
            metadata_filter: Optional metadata filter passed to Chroma as `where`.

        Returns:
            Retrieved chunk dictionaries with metadata and similarity score.

        Raises:
            ValueError: If query is empty or top_k is not positive.
        """
        if query is None or not query.strip():
            raise ValueError("query must be a non-empty string")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if self.count() == 0:
            return []

        query_embedding = self.embedding_model.embed_text(query.strip())
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if metadata_filter is not None:
            query_kwargs["where"] = metadata_filter

        results = self.collection.query(**query_kwargs)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted_results: list[dict] = []

        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            score = None if distance is None else 1.0 / (1.0 + float(distance))

            page_number = metadata.get("page_number")
            chunk_index = metadata.get("chunk_index")

            formatted_results.append(
                {
                    "text": document,
                    "document_id": metadata.get("document_id") or None,
                    "source_file": metadata.get("source_file") or None,
                    "page_number": None if page_number == -1 else page_number,
                    "chunk_id": metadata.get("chunk_id"),
                    "chunk_index": None if chunk_index == -1 else chunk_index,
                    "score": score,
                }
            )

        return formatted_results

    def count(self) -> int:
        """
        Return the number of documents stored in the collection.
        """
        return int(self.collection.count())

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks associated with a document identifier."""
        if document_id is None or not str(document_id).strip():
            raise ValueError("document_id cannot be empty")

        self.collection.delete(
            where={"document_id": str(document_id).strip()},
        )

    def clear(self) -> None:
        """
        Remove all documents from the collection.
        """
        current_count = self.count()
        if current_count == 0:
            return

        existing = self.collection.get(limit=current_count)
        ids = existing.get("ids", [])

        if ids:
            self.collection.delete(ids=ids)

    def _resolve_chunk_id(self, document: dict, index: int) -> str:
        """
        Resolve a stable chunk identifier for storage.
        """
        chunk_id = document.get("chunk_id")
        if chunk_id is not None and str(chunk_id).strip():
            return str(chunk_id).strip()

        source_file = str(document.get("source_file") or "unknown_source")
        page_number = document.get("page_number")
        chunk_index = document.get("chunk_index")

        page_label = page_number if page_number is not None else "None"
        chunk_label = chunk_index if chunk_index is not None else index
        return f"{source_file}_p{page_label}_c{chunk_label}"

    def _build_metadata(self, document: dict, chunk_id: str) -> dict:
        """
        Build Chroma-compatible primitive metadata for a document.
        """
        source_file = str(document.get("source_file") or "unknown_source")
        page_number = document.get("page_number")
        chunk_index = document.get("chunk_index")

        metadata = {
            "source_file": source_file,
            "page_number": int(page_number) if page_number is not None else -1,
            "chunk_id": chunk_id,
            "chunk_index": int(chunk_index) if chunk_index is not None else -1,
        }

        document_id = document.get("document_id")
        if document_id is not None and str(document_id).strip():
            metadata["document_id"] = str(document_id).strip()

        return metadata

    def _validate_text(self, text: str | None, error_message: str) -> str:
        """
        Validate and normalize a stored document text value.
        """
        if text is None or not str(text).strip():
            raise ValueError(error_message)

        return str(text).strip()

    def _validate_embedding(self, embedding) -> list[float]:
        """
        Validate and normalize an externally provided embedding vector.
        """
        if not isinstance(embedding, list):
            raise ValueError("embedding must be a list of numeric values")

        if not embedding:
            raise ValueError("embedding cannot be empty")

        if not all(isinstance(value, (int, float)) for value in embedding):
            raise ValueError("embedding must contain only numeric values")

        return [float(value) for value in embedding]
