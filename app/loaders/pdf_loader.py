from pathlib import Path


def load_pdf_text(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # Placeholder only: real PDF parsing will be added later.
    return ""