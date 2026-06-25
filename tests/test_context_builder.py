import pytest

from app.rag.context_builder import build_context, extract_sources


def test_build_context_returns_empty_string_for_empty_input():
    assert build_context([]) == ""


def test_build_context_raises_value_error_for_non_positive_max_chars():
    with pytest.raises(ValueError):
        build_context([{"text": "sample"}], max_chars=0)


def test_build_context_includes_expected_fields_for_single_chunk():
    retrieved_chunks = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "source_file": "sample.pdf",
            "page_number": 1,
            "score": 0.91234,
            "text": "Revenue increased by 12.5%",
        }
    ]

    context = build_context(retrieved_chunks)

    assert "[Source 1]" in context
    assert "source_file: sample.pdf" in context
    assert "page_number: 1" in context
    assert "chunk_id: sample.pdf_p1_c0" in context
    assert "score: 0.9123" in context
    assert "content:\nRevenue increased by 12.5%" in context


def test_build_context_preserves_source_order_for_multiple_chunks():
    retrieved_chunks = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "source_file": "sample.pdf",
            "page_number": 1,
            "score": 0.91,
            "text": "First chunk text",
        },
        {
            "chunk_id": "sample.pdf_p2_c0",
            "source_file": "sample.pdf",
            "page_number": 2,
            "score": 0.82,
            "text": "Second chunk text",
        },
    ]

    context = build_context(retrieved_chunks)

    assert context.index("[Source 1]") < context.index("[Source 2]")


def test_build_context_normalizes_whitespace_in_text():
    retrieved_chunks = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "source_file": "sample.pdf",
            "page_number": 1,
            "score": 0.91,
            "text": "Revenue\n\n   increased\t\tby 12.5%   in Q1",
        }
    ]

    context = build_context(retrieved_chunks)

    assert "content:\nRevenue increased by 12.5% in Q1" in context


def test_build_context_truncates_when_max_chars_is_small():
    retrieved_chunks = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "source_file": "sample.pdf",
            "page_number": 1,
            "score": 0.91,
            "text": "A" * 500,
        }
    ]

    context = build_context(retrieved_chunks, max_chars=120)

    assert "[Context truncated due to max_chars limit]" in context
    assert len(context) <= 120


def test_extract_sources_returns_expected_fields():
    retrieved_chunks = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "source_file": "sample.pdf",
            "page_number": 1,
            "score": 0.91,
            "text": "Revenue increased by 12.5%",
        }
    ]

    sources = extract_sources(retrieved_chunks)

    assert sources == [
        {
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_id": "sample.pdf_p1_c0",
            "score": 0.91,
        }
    ]


def test_extract_sources_does_not_repeat_duplicate_sources():
    retrieved_chunks = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "source_file": "sample.pdf",
            "page_number": 1,
            "score": 0.91,
            "text": "First occurrence",
        },
        {
            "chunk_id": "sample.pdf_p1_c0",
            "source_file": "sample.pdf",
            "page_number": 1,
            "score": 0.91,
            "text": "Duplicate occurrence",
        },
    ]

    sources = extract_sources(retrieved_chunks)

    assert len(sources) == 1
    assert sources[0]["chunk_id"] == "sample.pdf_p1_c0"


def test_extract_sources_does_not_fail_when_score_is_missing():
    retrieved_chunks = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "source_file": "sample.pdf",
            "page_number": 1,
            "text": "Revenue increased by 12.5%",
        }
    ]

    sources = extract_sources(retrieved_chunks)

    assert sources == [
        {
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_id": "sample.pdf_p1_c0",
            "score": None,
        }
    ]
