from app.processing.chunking import chunk_text


def test_chunk_text_returns_chunks():
    text = "a" * 120
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) >= 2
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_chunk_text_empty_input():
    assert chunk_text("") == []