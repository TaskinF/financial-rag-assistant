import pytest

from app.api.schemas import DocumentAskRequest
from app.services.rag_service import RAGService


class FakeDocumentRegistry:
    def __init__(self, documents: dict[str, dict] | None = None) -> None:
        self.documents = documents or {}

    def get_document(self, document_id: str) -> dict | None:
        return self.documents.get(document_id)


class FakeChromaVectorStore:
    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.search_calls: list[dict] = []

    def similarity_search(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        self.search_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "metadata_filter": metadata_filter,
            }
        )
        return self.results[:top_k]


def build_results() -> list[dict]:
    return [
        {
            "text": "Management fee is 7,125,856.54 TL.",
            "document_id": "doc_a",
            "source_file": "doc_a.pdf",
            "page_number": 3,
            "chunk_id": "doc_a.pdf_p3_c0",
            "chunk_index": 0,
            "score": 0.95,
        },
        {
            "text": "The management fee ratio is 0.1836 percent.",
            "document_id": "doc_a",
            "source_file": "doc_a.pdf",
            "page_number": 3,
            "chunk_id": "doc_a.pdf_p3_c1",
            "chunk_index": 1,
            "score": 0.90,
        },
        {
            "text": "Portfolio value is listed on the same report.",
            "document_id": "doc_a",
            "source_file": "doc_a.pdf",
            "page_number": 4,
            "chunk_id": "doc_a.pdf_p4_c0",
            "chunk_index": 0,
            "score": 0.80,
        },
    ]


def build_service(results: list[dict] | None = None) -> tuple[RAGService, FakeChromaVectorStore]:
    service = RAGService()
    service._document_registry = FakeDocumentRegistry(
        {"doc_a": {"document_id": "doc_a"}}
    )
    fake_store = FakeChromaVectorStore(results if results is not None else build_results())
    service._chroma_vector_store = fake_store
    return service, fake_store


def test_answer_document_question_raises_for_missing_document_id():
    service = RAGService()
    service._document_registry = FakeDocumentRegistry()

    with pytest.raises(FileNotFoundError):
        service.answer_document_question(
            "missing_doc",
            DocumentAskRequest(question="What is the management fee?"),
        )


def test_answer_document_question_uses_document_metadata_filter():
    service, fake_store = build_service()
    request = DocumentAskRequest(
        question="What is the management fee?",
        top_k=3,
        answer_top_k=2,
    )

    service.answer_document_question("doc_a", request)

    assert fake_store.search_calls == [
        {
            "query": "What is the management fee?",
            "top_k": 3,
            "metadata_filter": {"document_id": "doc_a"},
        }
    ]


def test_answer_document_question_limits_context_and_sources_to_answer_top_k():
    service, _ = build_service()
    request = DocumentAskRequest(
        question="What is the management fee?",
        top_k=3,
        answer_top_k=1,
    )

    response = service.answer_document_question("doc_a", request)

    assert response.retrieved_count == 3
    assert response.answer_context_count == 1
    assert len(response.sources) == 1
    assert response.sources[0].chunk_id == "doc_a.pdf_p3_c0"


def test_answer_document_question_uses_fake_provider_and_returns_response_fields():
    service, _ = build_service()
    request = DocumentAskRequest(
        question="What is the management fee?",
        llm_provider="fake",
    )

    response = service.answer_document_question("doc_a", request)

    assert response.document_id == "doc_a"
    assert response.answer == "Fake answer preview"
    assert response.sources
    assert response.retrieved_count == 3
    assert response.answer_context_count == 2
    assert response.llm_provider == "fake"


def test_answer_document_question_uses_mocked_ollama_client(monkeypatch):
    created_clients: list[dict] = []

    class MockOllamaLLMClient:
        def __init__(self, base_url: str, model: str) -> None:
            created_clients.append({"base_url": base_url, "model": model})

        def generate(self, prompt: str) -> str:
            return "Mock Ollama answer"

    monkeypatch.setattr(
        "app.services.rag_service.OllamaLLMClient",
        MockOllamaLLMClient,
    )
    service, _ = build_service()
    request = DocumentAskRequest(
        question="What is the management fee?",
        llm_provider="ollama",
        llm_model="gemma3:4b",
    )

    response = service.answer_document_question("doc_a", request)

    assert response.answer == "Mock Ollama answer"
    assert response.llm_model == "gemma3:4b"
    assert created_clients[0]["model"] == "gemma3:4b"


def test_answer_document_question_supports_empty_retrieval_results():
    service, _ = build_service(results=[])

    response = service.answer_document_question(
        "doc_a",
        DocumentAskRequest(question="Is this information available?"),
    )

    assert response.answer == "Fake answer preview"
    assert response.sources == []
    assert response.retrieved_count == 0
    assert response.answer_context_count == 0
