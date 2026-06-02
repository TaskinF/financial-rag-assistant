def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text.strip():
        return []

    chunks = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break

    return chunks