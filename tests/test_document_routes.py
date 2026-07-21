from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from requests.exceptions import ConnectionError

from app.api import document_routes
from app.api.document_routes import (
    get_document_indexing_service,
    get_rag_service,
)
from app.main import app


def document_record(document_id: str = "doc_a") -> dict:
    return {
        "document_id": document_id,
        "filename": f"{document_id}.pdf",
        "path": f"data/documents/{document_id}.pdf",
        "chunk_count": 2,
        "collection_name": "financial_documents",
        "indexed_at": "2026-07-21T12:00:00Z",
        "status": "indexed",
    }


class FakeDocumentIndexingService:
    def __init__(self, documents: list[dict] | None = None) -> None:
        self.documents = documents or []
        self.index_calls: list[dict] = []
        self.temporary_file_existed_during_call = False

    def list_documents(self) -> list[dict]:
        return self.documents

    def index_pdf(self, **kwargs) -> dict:
        temporary_path = Path(kwargs["pdf_path"])
        self.temporary_file_existed_during_call = temporary_path.exists()
        self.index_calls.append(kwargs)
        return document_record(kwargs.get("document_id") or "generated_doc")


class FakeRAGService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, object]] = []

    def answer_document_question(self, document_id: str, request):
        self.calls.append((document_id, request))
        if self.error is not None:
            raise self.error

        return {
            "document_id": document_id,
            "question": request.question,
            "answer": "Fake answer preview",
            "sources": [
                {
                    "source_file": f"{document_id}.pdf",
                    "page_number": 3,
                    "chunk_id": f"{document_id}.pdf_p3_c0",
                    "score": 0.95,
                }
            ],
            "retrieved_count": 2,
            "answer_context_count": 1,
            "llm_provider": request.llm_provider,
            "llm_model": request.llm_model,
        }


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def override_indexing_service(service: FakeDocumentIndexingService) -> None:
    app.dependency_overrides[get_document_indexing_service] = lambda: service


def override_rag_service(service: FakeRAGService) -> None:
    app.dependency_overrides[get_rag_service] = lambda: service


def test_list_documents_returns_empty_response(client):
    override_indexing_service(FakeDocumentIndexingService())

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {"documents": [], "total": 0}


def test_list_documents_returns_registered_documents(client):
    override_indexing_service(
        FakeDocumentIndexingService(documents=[document_record()])
    )

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["documents"][0]["document_id"] == "doc_a"


def test_valid_pdf_upload_returns_document_metadata(client):
    service = FakeDocumentIndexingService()
    override_indexing_service(service)

    response = client.post(
        "/documents/upload",
        files={"file": ("report.pdf", b"%PDF-1.4 test", "application/pdf")},
        data={"document_id": "doc_a"},
    )

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc_a"
    assert response.json()["filename"] == "doc_a.pdf"
    assert response.json()["chunk_count"] == 2


def test_upload_rejects_non_pdf_extension(client):
    override_indexing_service(FakeDocumentIndexingService())

    response = client.post(
        "/documents/upload",
        files={"file": ("report.txt", b"not a pdf", "application/pdf")},
    )

    assert response.status_code == 400


def test_upload_rejects_non_pdf_content_type(client):
    override_indexing_service(FakeDocumentIndexingService())

    response = client.post(
        "/documents/upload",
        files={"file": ("report.pdf", b"not a pdf", "text/plain")},
    )

    assert response.status_code == 400


def test_upload_rejects_file_larger_than_50_mb(client, monkeypatch):
    assert document_routes.MAX_UPLOAD_SIZE == 50 * 1024 * 1024
    monkeypatch.setattr(document_routes, "MAX_UPLOAD_SIZE", 8)
    override_indexing_service(FakeDocumentIndexingService())

    response = client.post(
        "/documents/upload",
        files={"file": ("large.pdf", b"123456789", "application/pdf")},
    )

    assert response.status_code == 413


def test_upload_passes_parameters_to_indexing_service(client):
    service = FakeDocumentIndexingService()
    override_indexing_service(service)

    response = client.post(
        "/documents/upload",
        files={"file": ("report.pdf", b"%PDF-1.4 test", "application/pdf")},
        data={
            "document_id": "doc_a",
            "chunk_size": "800",
            "chunk_overlap": "100",
            "start_page": "2",
            "end_page": "5",
        },
    )

    assert response.status_code == 200
    call = service.index_calls[0]
    assert call["document_id"] == "doc_a"
    assert call["chunk_size"] == 800
    assert call["chunk_overlap"] == 100
    assert call["start_page"] == 2
    assert call["end_page"] == 5


def test_upload_removes_temporary_file_after_indexing(client):
    service = FakeDocumentIndexingService()
    override_indexing_service(service)

    response = client.post(
        "/documents/upload",
        files={"file": ("report.pdf", b"%PDF-1.4 test", "application/pdf")},
    )

    temporary_path = Path(service.index_calls[0]["pdf_path"])
    assert response.status_code == 200
    assert service.temporary_file_existed_during_call is True
    assert not temporary_path.exists()


def test_ask_document_returns_answer_sources_and_document_id(client):
    service = FakeRAGService()
    override_rag_service(service)

    response = client.post(
        "/documents/doc_a/ask",
        json={"question": "What is the management fee?"},
    )

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc_a"
    assert response.json()["answer"] == "Fake answer preview"
    assert response.json()["sources"][0]["chunk_id"] == "doc_a.pdf_p3_c0"


def test_ask_document_returns_404_for_missing_document(client):
    override_rag_service(FakeRAGService(error=FileNotFoundError("missing")))

    response = client.post(
        "/documents/missing_doc/ask",
        json={"question": "What is the management fee?"},
    )

    assert response.status_code == 404


def test_ask_document_returns_422_for_invalid_request_body(client):
    override_rag_service(FakeRAGService())

    response = client.post(
        "/documents/doc_a/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_ask_document_returns_503_when_ollama_is_unreachable(client):
    override_rag_service(
        FakeRAGService(error=ConnectionError("connection refused"))
    )

    response = client.post(
        "/documents/doc_a/ask",
        json={
            "question": "What is the management fee?",
            "llm_provider": "ollama",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Ollama is not reachable. Make sure Ollama is running and "
        "the selected model is available."
    )


def test_health_endpoint_remains_available(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
