from app.loaders.pdf_loader import load_pdf_pages
from app.processing.chunking import chunk_pages
from app.processing.text_cleaner import clean_text


def build_chunks_from_pdf(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Load a PDF, clean each page's text, and build metadata-preserving chunks.

    Args:
        file_path: Path to the PDF file.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of chunk dictionaries containing chunk text and metadata.
    """
    pages = load_pdf_pages(file_path)

    cleaned_pages: list[dict] = []

    for page in pages:
        cleaned_pages.append(
            {
                "text": clean_text(page.get("text")),
                "source_file": page.get("source_file"),
                "page_number": page.get("page_number"),
            }
        )

    return chunk_pages(
        cleaned_pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )