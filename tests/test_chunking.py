import pytest

from app.processing.chunking import chunk_pages, chunk_text


def test_chunk_text_returns_empty_list_for_empty_string():
    assert chunk_text("") == []


def test_chunk_text_returns_empty_list_for_none_input():
    assert chunk_text(None) == []


def test_chunk_text_raises_value_error_when_chunk_size_is_not_positive():
    with pytest.raises(ValueError):
        chunk_text("sample text", chunk_size=0)


def test_chunk_text_raises_value_error_when_chunk_overlap_is_negative():
    with pytest.raises(ValueError):
        chunk_text("sample text", chunk_overlap=-1)


def test_chunk_text_raises_value_error_when_chunk_overlap_is_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("sample text", chunk_size=10, chunk_overlap=10)


def test_chunk_text_returns_single_chunk_for_short_text():
    text = "Short financial note."
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)

    assert chunks == ["Short financial note."]


def test_chunk_text_returns_multiple_chunks_for_long_text():
    text = "abcdefghij" * 3
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)

    assert len(chunks) > 1


def test_chunk_text_applies_overlap_correctly():
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=4, chunk_overlap=1)

    assert chunks == ["abcd", "defg", "ghij"]


def test_chunk_pages_preserves_metadata_fields():
    pages = [
        {
            "text": "abcdefghij",
            "source_file": "sample.pdf",
            "page_number": 1,
        }
    ]

    chunks = chunk_pages(pages, chunk_size=4, chunk_overlap=1)

    assert chunks[0]["source_file"] == "sample.pdf"
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["chunk_id"] == "sample.pdf_p1_c0"


def test_chunk_pages_generates_deterministic_chunk_ids():
    pages = [
        {
            "text": "abcdefghij",
            "source_file": "sample.pdf",
            "page_number": 1,
        }
    ]

    chunks = chunk_pages(pages, chunk_size=4, chunk_overlap=1)

    assert chunks[0]["chunk_id"] == "sample.pdf_p1_c0"
    assert chunks[1]["chunk_id"] == "sample.pdf_p1_c1"


def test_chunk_pages_uses_unknown_source_when_source_file_is_missing():
    pages = [
        {
            "text": "abcdef",
            "page_number": 2,
        }
    ]

    chunks = chunk_pages(pages, chunk_size=4, chunk_overlap=1)

    assert chunks[0]["source_file"] == "unknown_source"
    assert chunks[0]["chunk_id"] == "unknown_source_p2_c0"


def test_chunk_pages_uses_none_when_page_number_is_missing():
    pages = [
        {
            "text": "abcdef",
            "source_file": "sample.pdf",
        }
    ]

    chunks = chunk_pages(pages, chunk_size=4, chunk_overlap=1)

    assert chunks[0]["page_number"] is None
    assert chunks[0]["chunk_id"] == "sample.pdf_pNone_c0"