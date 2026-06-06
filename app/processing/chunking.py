def chunk_text(
    text: str | None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    Args:
        text: Input text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of non-empty, stripped text chunks.

    Raises:
        ValueError: If chunk_size is not positive, if chunk_overlap is negative,
            or if chunk_overlap is greater than or equal to chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if text is None:
        return []

    normalized_text = text.strip()
    if not normalized_text:
        return []

    chunks: list[str] = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(normalized_text), step):
        chunk = normalized_text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(normalized_text):
            break

    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Split page dictionaries into text chunks while preserving page metadata.

    Args:
        pages: List of page dictionaries returned by the PDF loader.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of chunk dictionaries containing chunk text and metadata.
    """
    chunked_pages: list[dict] = []

    for page in pages:
        text = page.get("text")
        source_file = page.get("source_file", "unknown_source")
        page_number = page.get("page_number", None)

        text_chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk in enumerate(text_chunks):
            chunked_pages.append(
                {
                    "chunk_id": f"{source_file}_p{page_number}_c{chunk_index}",
                    "text": chunk,
                    "source_file": source_file,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                }
            )

    return chunked_pages