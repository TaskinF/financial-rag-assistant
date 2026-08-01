from fastapi.testclient import TestClient
from requests.exceptions import ConnectionError

import app.main as main_module
from app.main import app


class FakeRAGService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def answer_question(self, request):
        if self.error is not None:
            raise self.error

        return {
            "question": request.question,
            "answer": "Fake answer preview",
            "sources": [],
            "retrieved_count": 0,
            "answer_context_count": 0,
            "llm_provider": request.llm_provider,
            "llm_model": "fake",
        }


def test_legacy_ask_endpoint_remains_available(monkeypatch):
    monkeypatch.setattr(main_module, "rag_service", FakeRAGService())

    response = TestClient(app).post(
        "/ask",
        json={"question": "What is the fund value?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Fake answer preview"


def test_legacy_ask_returns_404_for_missing_pdf(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "rag_service",
        FakeRAGService(error=FileNotFoundError("sensitive local path")),
    )

    response = TestClient(app).post(
        "/ask",
        json={"question": "What is the fund value?"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "PDF file not found"


def test_legacy_ask_returns_503_when_ollama_is_unreachable(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "rag_service",
        FakeRAGService(error=ConnectionError("connection refused")),
    )

    response = TestClient(app).post(
        "/ask",
        json={
            "question": "What is the fund value?",
            "llm_provider": "ollama",
        },
    )

    assert response.status_code == 503
