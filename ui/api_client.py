from pathlib import Path

import requests


class RAGAPIClient:
    """HTTP client for the Financial Document RAG API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: int = 300,
    ) -> None:
        """Initialize the API client."""
        if base_url is None or not str(base_url).strip():
            raise ValueError("base_url cannot be empty")

        normalized_base_url = str(base_url).strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("base_url cannot be empty")

        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        self.base_url = normalized_base_url
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> dict:
        """Return the API health status."""
        response = self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def list_documents(self) -> dict:
        """Return indexed documents from the API."""
        response = self.session.get(
            f"{self.base_url}/documents",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def upload_document(
        self,
        filename: str,
        file_bytes: bytes,
        document_id: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> dict:
        """Upload and index a PDF document."""
        if filename is None or not str(filename).strip():
            raise ValueError("filename cannot be empty")

        normalized_filename = str(filename).strip()
        if Path(normalized_filename).suffix.lower() != ".pdf":
            raise ValueError("filename must have a .pdf extension")

        if not file_bytes:
            raise ValueError("file_bytes cannot be empty")

        files = {
            "file": (
                normalized_filename,
                file_bytes,
                "application/pdf",
            )
        }
        form_values = {
            "document_id": document_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "start_page": start_page,
            "end_page": end_page,
        }
        data = {
            key: str(value)
            for key, value in form_values.items()
            if value is not None
        }

        response = self.session.post(
            f"{self.base_url}/documents/upload",
            files=files,
            data=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def ask_document(
        self,
        document_id: str,
        question: str,
        top_k: int = 3,
        answer_top_k: int = 2,
        llm_provider: str = "ollama",
        llm_model: str = "gemma3:4b",
        max_context_chars: int = 4000,
    ) -> dict:
        """Ask a question about one indexed document."""
        if document_id is None or not str(document_id).strip():
            raise ValueError("document_id cannot be empty")

        if question is None or not str(question).strip():
            raise ValueError("question cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if answer_top_k <= 0:
            raise ValueError("answer_top_k must be greater than 0")

        if answer_top_k > top_k:
            raise ValueError("answer_top_k cannot be greater than top_k")

        if llm_provider not in {"fake", "ollama"}:
            raise ValueError("llm_provider must be either 'fake' or 'ollama'")

        normalized_document_id = str(document_id).strip()
        payload = {
            "question": str(question).strip(),
            "top_k": top_k,
            "answer_top_k": answer_top_k,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "max_context_chars": max_context_chars,
        }

        response = self.session.post(
            f"{self.base_url}/documents/{normalized_document_id}/ask",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
