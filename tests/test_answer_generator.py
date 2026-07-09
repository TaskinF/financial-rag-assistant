import pytest

from app.llm.llm_client import FakeLLMClient
from app.rag.answer_generator import generate_answer


def test_generate_answer_raises_value_error_for_empty_question():
    with pytest.raises(ValueError):
        generate_answer(
            question="",
            retrieved_chunks=[],
            llm_client=FakeLLMClient(),
        )


def test_generate_answer_raises_value_error_for_none_llm_client():
    with pytest.raises(ValueError):
        generate_answer(
            question="Fonun toplam degeri nedir?",
            retrieved_chunks=[],
            llm_client=None,
        )


def test_generate_answer_raises_value_error_for_non_positive_max_context_chars():
    with pytest.raises(ValueError):
        generate_answer(
            question="Fonun toplam degeri nedir?",
            retrieved_chunks=[],
            llm_client=FakeLLMClient(),
            max_context_chars=0,
        )


def test_generate_answer_returns_expected_payload():
    retrieved_chunks = [
        {
            "chunk_id": "KPC_2026.05.pdf_p3_c0",
            "source_file": "KPC_2026.05.pdf",
            "page_number": 3,
            "score": 0.91,
            "text": "Fon toplam degeri 3.881.770.473,07 olarak gecmektedir.",
        }
    ]

    result = generate_answer(
        question="Fonun toplam degeri nedir?",
        retrieved_chunks=retrieved_chunks,
        llm_client=FakeLLMClient(response="Fon toplam degeri 3.881.770.473,07'dir."),
        max_context_chars=4000,
    )

    assert result["question"] == "Fonun toplam degeri nedir?"
    assert result["answer"] == "Fon toplam degeri 3.881.770.473,07'dir."
    assert result["sources"] == [
        {
            "source_file": "KPC_2026.05.pdf",
            "page_number": 3,
            "chunk_id": "KPC_2026.05.pdf_p3_c0",
            "score": 0.91,
        }
    ]
    assert "KPC_2026.05.pdf" in result["context"]
    assert "Fonun toplam degeri nedir?" in result["prompt"]


def test_generate_answer_cleans_prefixed_llm_output():
    result = generate_answer(
        question="Fonun yonetim ucreti nedir?",
        retrieved_chunks=[],
        llm_client=FakeLLMClient(response="Cevap: Fon yonetim ucreti vardir."),
    )

    assert result["answer"] == "Fon yonetim ucreti vardir."
