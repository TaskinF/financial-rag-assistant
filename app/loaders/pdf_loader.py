from pathlib import Path

import fitz


def load_pdf_pages(file_path: str) -> list[dict]:
    """
    Load a PDF file page by page and return non-empty page texts.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of dictionaries containing page_number, text, and source_file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    pages: list[dict] = []

    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text().strip()

            if not text:
                continue

            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "source_file": str(path),
                }
            )

    return pages