import pytest

from app.rag.matched_lines import extract_matched_lines


def build_retrieved_chunks() -> list[dict]:
    return [
        {
            "source_file": "KPC_2026.05.pdf",
            "page_number": 3,
            "chunk_id": "KPC_2026.05.pdf_p3_c1",
            "score": 0.6254,
            "text": (
                "Fon Toplam Degeri 3.880.911.776,55\n"
                "Fon Yonetim Ucreti 7.125.856,54 0,1836 %\n"
                "Portfoy Degeri 3.912.000.000,00"
            ),
        },
        {
            "source_file": "KPC_2026.05.pdf",
            "page_number": 4,
            "chunk_id": "KPC_2026.05.pdf_p4_c0",
            "score": 0.5111,
            "text": (
                "Fonun amaci uzun vadeli getiri saglamaktir.\n"
                "Yonetim ekibi piyasa kosullarini takip eder."
            ),
        },
    ]


def test_extract_matched_lines_raises_value_error_for_empty_question():
    with pytest.raises(ValueError):
        extract_matched_lines("", build_retrieved_chunks())


def test_extract_matched_lines_raises_value_error_for_non_positive_max_lines():
    with pytest.raises(ValueError):
        extract_matched_lines(
            "Fonun yonetim ucreti nedir?",
            build_retrieved_chunks(),
            max_lines=0,
        )


def test_extract_matched_lines_returns_empty_list_for_empty_retrieved_chunks():
    assert extract_matched_lines("Fonun yonetim ucreti nedir?", []) == []


def test_extract_matched_lines_returns_management_fee_line():
    results = extract_matched_lines(
        "Fonun yonetim ucreti nedir?",
        build_retrieved_chunks(),
    )

    assert any(
        item["line"] == "Fon Yonetim Ucreti 7.125.856,54 0,1836 %"
        for item in results
    )


def test_extract_matched_lines_result_contains_expected_fields():
    results = extract_matched_lines(
        "Fonun yonetim ucreti nedir?",
        build_retrieved_chunks(),
    )

    first_item = results[0]

    assert "source_file" in first_item
    assert "page_number" in first_item
    assert "chunk_id" in first_item
    assert "score" in first_item
    assert "match_score" in first_item


def test_extract_matched_lines_respects_max_lines_limit():
    results = extract_matched_lines(
        "Fon nedir?",
        build_retrieved_chunks(),
        max_lines=1,
    )

    assert len(results) == 1


def test_extract_matched_lines_does_not_return_duplicate_lines():
    duplicated_chunks = [
        {
            "source_file": "KPC_2026.05.pdf",
            "page_number": 3,
            "chunk_id": "KPC_2026.05.pdf_p3_c1",
            "score": 0.6254,
            "text": "Fon Yonetim Ucreti 7.125.856,54 0,1836 %",
        },
        {
            "source_file": "KPC_2026.05.pdf",
            "page_number": 5,
            "chunk_id": "KPC_2026.05.pdf_p5_c0",
            "score": 0.5000,
            "text": "Fon Yonetim Ucreti 7.125.856,54 0,1836 %",
        },
    ]

    results = extract_matched_lines(
        "Fonun yonetim ucreti nedir?",
        duplicated_chunks,
        max_lines=5,
    )

    matching_lines = [
        item
        for item in results
        if item["line"] == "Fon Yonetim Ucreti 7.125.856,54 0,1836 %"
    ]

    assert len(matching_lines) == 1


def test_financial_value_line_scores_higher_than_non_financial_line():
    chunks = [
        {
            "source_file": "KPC_2026.05.pdf",
            "page_number": 3,
            "chunk_id": "KPC_2026.05.pdf_p3_c1",
            "score": 0.6254,
            "text": (
                "Fon Yonetim Ucreti 7.125.856,54 0,1836 %\n"
                "Yonetim yaklasimi piyasa kosullarina gore belirlenir."
            ),
        }
    ]

    results = extract_matched_lines(
        "Fonun yonetim ucreti nedir?",
        chunks,
        max_lines=5,
    )

    assert len(results) >= 2
    assert results[0]["line"] == "Fon Yonetim Ucreti 7.125.856,54 0,1836 %"
    assert results[0]["match_score"] > results[1]["match_score"]
