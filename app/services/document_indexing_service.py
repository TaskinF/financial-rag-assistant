from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.documents.document_registry import DocumentRegistry, create_document_id
from app.processing.ingestion import build_chunks_from_pdf
from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.embeddings import BGEFlagEmbeddingModel


class DocumentIndexingService:
    """
    Service for copying, chunking, indexing, and registering PDF documents.
    """

    def __init__(
        self,
        documents_dir: str = "data/documents",
        registry_path: str = "artifacts/document_registry.json",
        chroma_dir: str = "artifacts/chroma",
        collection_name: str = "financial_documents",
    ) -> None:
        """
        Initialize indexing service dependencies with lazy vector components.
        """
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        self.registry_path = registry_path
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name

        self.registry = DocumentRegistry(registry_path=registry_path)
        self._embedding_model: BGEFlagEmbeddingModel | None = None
        self._vector_store: ChromaVectorStore | None = None

    def _get_embedding_model(self):
        """
        Lazily create the embedding model.
        """
        if self._embedding_model is None:
            self._embedding_model = BGEFlagEmbeddingModel(use_fp16=False)

        return self._embedding_model

    def _get_vector_store(self):
        """
        Lazily create the Chroma vector store.
        """
        if self._vector_store is None:
            self._vector_store = ChromaVectorStore(
                embedding_model=self._get_embedding_model(),
                persist_dir=self.chroma_dir,
                collection_name=self.collection_name,
            )

        return self._vector_store

    def index_pdf(
        self,
        pdf_path: str,
        document_id: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> dict:
        """
        Copy a PDF into the managed documents directory and index its chunks.
        """
        if pdf_path is None or not str(pdf_path).strip():
            raise ValueError("pdf_path cannot be empty")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        if start_page is not None and start_page <= 0:
            raise ValueError("start_page must be greater than 0")

        if end_page is not None and end_page <= 0:
            raise ValueError("end_page must be greater than 0")

        if start_page is not None and end_page is not None and start_page > end_page:
            raise ValueError("start_page cannot be greater than end_page")

        source_path = Path(str(pdf_path).strip())
        if not source_path.exists():
            raise FileNotFoundError(f"PDF file not found: {source_path}")

        if source_path.suffix.lower() != ".pdf":
            raise ValueError("pdf_path must point to a .pdf file")

        if document_id is None:
            resolved_document_id = create_document_id(source_path.name)
        else:
            resolved_document_id = str(document_id).strip()
            if not resolved_document_id:
                raise ValueError("document_id cannot be empty")

        saved_pdf_path = self.documents_dir / f"{resolved_document_id}.pdf"
        shutil.copy2(source_path, saved_pdf_path)

        chunks = build_chunks_from_pdf(
            file_path=str(saved_pdf_path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = self._filter_chunks_by_page(chunks, start_page=start_page, end_page=end_page)

        if not chunks:
            raise ValueError("no chunks remain after page filtering")

        normalized_chunks: list[dict] = []
        for chunk in chunks:
            normalized_chunk = dict(chunk)
            normalized_chunk["document_id"] = resolved_document_id
            normalized_chunk["source_file"] = saved_pdf_path.name
            normalized_chunks.append(normalized_chunk)

        vector_store = self._get_vector_store()
        vector_store.delete_by_document_id(resolved_document_id)
        vector_store.add_documents(normalized_chunks)

        indexed_at = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

        registry_record = {
            "document_id": resolved_document_id,
            "filename": saved_pdf_path.name,
            "path": str(saved_pdf_path),
            "chunk_count": len(normalized_chunks),
            "collection_name": self.collection_name,
            "indexed_at": indexed_at,
            "status": "indexed",
        }

        return self.registry.register_document(registry_record)

    def list_documents(self) -> list[dict]:
        """
        Return all registered documents.
        """
        return self.registry.list_documents()

    def get_document(self, document_id: str) -> dict | None:
        """
        Return a single document by document_id.
        """
        return self.registry.get_document(document_id)

    def _filter_chunks_by_page(
        self,
        chunks: list[dict],
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> list[dict]:
        """
        Filter chunk list by optional page range.
        """
        filtered_chunks: list[dict] = []

        for chunk in chunks:
            page_number = chunk.get("page_number")

            if start_page is not None and page_number is not None and page_number < start_page:
                continue

            if end_page is not None and page_number is not None and page_number > end_page:
                continue

            filtered_chunks.append(chunk)

        return filtered_chunks
