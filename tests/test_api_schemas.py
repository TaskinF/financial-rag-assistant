import pytest
from pydantic import ValidationError

from app.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentAskRequest,
    DocumentAskResponse,
    DocumentListResponse,
    DocumentResponse,
)


def test_ask_request_can_be_created_with_valid_input():
    request = AskRequest(question="Fonun toplam değeri nedir?")

    assert request.question == "Fonun toplam değeri nedir?"
    assert request.top_k == 3
    assert request.answer_top_k == 2
    assert request.llm_provider == "fake"


def test_ask_request_raises_validation_error_for_empty_question():
    with pytest.raises(ValidationError):
        AskRequest(question="   ")


def test_ask_request_raises_validation_error_for_non_positive_top_k():
    with pytest.raises(ValidationError):
        AskRequest(question="Fonun toplam değeri nedir?", top_k=0)


def test_ask_request_raises_validation_error_for_non_positive_answer_top_k():
    with pytest.raises(ValidationError):
        AskRequest(question="Fonun toplam değeri nedir?", answer_top_k=0)


def test_ask_request_raises_validation_error_when_chunk_overlap_is_not_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        AskRequest(
            question="Fonun toplam değeri nedir?",
            chunk_size=500,
            chunk_overlap=500,
        )


def test_ask_request_raises_validation_error_for_invalid_llm_provider():
    with pytest.raises(ValidationError):
        AskRequest(
            question="Fonun toplam değeri nedir?",
            llm_provider="openai",
        )


def test_ask_response_can_be_created_with_valid_sources():
    response = AskResponse(
        question="Fonun toplam değeri nedir?",
        answer="Cevap: Fon toplam değeri 3.881.770.473,07 olarak geçmektedir.",
        sources=[
            {
                "source_file": "KPC_2026.05.pdf",
                "page_number": 3,
                "chunk_id": "KPC_2026.05.pdf_p3_c0",
                "score": 0.9123,
            }
        ],
        retrieved_count=3,
        answer_context_count=2,
        llm_provider="fake",
        llm_model="fake",
    )

    assert response.question == "Fonun toplam değeri nedir?"
    assert response.sources[0].source_file == "KPC_2026.05.pdf"
    assert response.sources[0].page_number == 3


def test_document_response_can_be_created_with_valid_input():
    response = DocumentResponse(
        document_id="kpc_2026_05_b02bb78b",
        filename="kpc_2026_05_b02bb78b.pdf",
        path="data/documents/kpc_2026_05_b02bb78b.pdf",
        chunk_count=10,
        collection_name="financial_documents",
        indexed_at="2026-07-21T10:30:00+00:00",
        status="indexed",
    )

    assert response.document_id == "kpc_2026_05_b02bb78b"
    assert response.chunk_count == 10


def test_document_list_response_can_be_created_with_valid_documents():
    document = DocumentResponse(
        document_id="kpc_document",
        filename="kpc_document.pdf",
        path="data/documents/kpc_document.pdf",
        chunk_count=5,
        collection_name="financial_documents",
        indexed_at="2026-07-21T10:30:00+00:00",
        status="indexed",
    )

    response = DocumentListResponse(documents=[document], total=1)

    assert response.total == 1
    assert response.documents[0].document_id == "kpc_document"


def test_document_ask_request_can_be_created_with_valid_input():
    request = DocumentAskRequest(question="What is the fund management fee?")

    assert request.question == "What is the fund management fee?"
    assert request.top_k == 3
    assert request.answer_top_k == 2
    assert request.llm_provider == "fake"


def test_document_ask_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        DocumentAskRequest(question="   ")


def test_document_ask_request_rejects_non_positive_top_k():
    with pytest.raises(ValidationError):
        DocumentAskRequest(question="What is the fund value?", top_k=0)


def test_document_ask_request_rejects_non_positive_answer_top_k():
    with pytest.raises(ValidationError):
        DocumentAskRequest(question="What is the fund value?", answer_top_k=0)


def test_document_ask_request_rejects_answer_top_k_greater_than_top_k():
    with pytest.raises(ValidationError):
        DocumentAskRequest(
            question="What is the fund value?",
            top_k=2,
            answer_top_k=3,
        )


def test_document_ask_request_rejects_invalid_llm_provider():
    with pytest.raises(ValidationError):
        DocumentAskRequest(
            question="What is the fund value?",
            llm_provider="openai",
        )


def test_document_ask_request_rejects_non_positive_max_context_chars():
    with pytest.raises(ValidationError):
        DocumentAskRequest(
            question="What is the fund value?",
            max_context_chars=0,
        )


def test_document_ask_response_can_be_created_with_valid_sources():
    response = DocumentAskResponse(
        document_id="kpc_document",
        question="What is the fund management fee?",
        answer="The fund management fee is 7,125,856.54 TL.",
        sources=[
            {
                "source_file": "kpc_document.pdf",
                "page_number": 3,
                "chunk_id": "kpc_document.pdf_p3_c0",
                "score": 0.9123,
            }
        ],
        retrieved_count=3,
        answer_context_count=2,
        llm_provider="fake",
        llm_model="fake",
    )

    assert response.document_id == "kpc_document"
    assert response.sources[0].chunk_id == "kpc_document.pdf_p3_c0"
