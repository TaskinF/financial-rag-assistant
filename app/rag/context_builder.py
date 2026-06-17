import re


def _normalize_text(text: str) -> str:
    """
    Normalize whitespace in text to single spaces.
    """
    return re.sub(r"\s+", " ", text).strip()


def build_context(retrieved_chunks: list[dict], max_chars: int = 4000) -> str:
    """
    Build an LLM-ready context string from retrieved chunks.

    Args:
        retrieved_chunks: Retrieved chunk dictionaries.
        max_chars: Maximum allowed context length.

    Returns:
        A formatted context string.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    if not retrieved_chunks:
        return ""

    sections: list[str] = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        source_file = chunk.get("source_file", "unknown_source")
        page_number = chunk.get("page_number")
        text = _normalize_text(str(chunk.get("text", "")))

        lines = [
            f"[Source {index}]",
            f"source_file: {source_file}",
            f"page_number: {page_number}",
        ]

        if "score" in chunk and chunk["score"] is not None:
            lines.append(f"score: {chunk['score']:.4f}")

        lines.append(f"text: {text}")
        sections.append("\n".join(lines))

    context = "\n\n".join(sections)

    if len(context) <= max_chars:
        return context

    truncation_note = "[Context truncated due to max_chars limit]"

    if max_chars <= len(truncation_note):
        return truncation_note[:max_chars]

    truncated_context = context[: max_chars - len(truncation_note)].rstrip()
    return f"{truncated_context}{truncation_note}"


def extract_sources(retrieved_chunks: list[dict]) -> list[dict]:
    """
    Extract unique source metadata from retrieved chunks while preserving order.

    Args:
        retrieved_chunks: Retrieved chunk dictionaries.

    Returns:
        A list of unique source metadata dictionaries.
    """
    sources: list[dict] = []
    seen: set[tuple] = set()

    for chunk in retrieved_chunks:
        source_file = chunk.get("source_file")
        page_number = chunk.get("page_number")
        chunk_id = chunk.get("chunk_id")

        key = (source_file, page_number, chunk_id)
        if key in seen:
            continue

        seen.add(key)

        source_info = {
            "source_file": source_file,
            "page_number": page_number,
            "chunk_id": chunk_id,
            "score": chunk.get("score"),
        }
        sources.append(source_info)

    return sources