import pytest
from pydantic import ValidationError

from app.api.schemas import AskRequest, AskResponse


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
