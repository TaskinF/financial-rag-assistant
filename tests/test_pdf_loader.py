import pytest
from pathlib import Path

from app.loaders.pdf_loader import load_pdf_pages


def test_load_pdf_pages_raises_file_not_found_for_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pdf_pages("missing_file.pdf")


def test_load_pdf_pages_raises_value_error_for_non_pdf_file():
    with pytest.raises(ValueError):
        load_pdf_pages("README.md")

def test_load_pdf_pages_uses_file_name_for_source_file(monkeypatch):
    class FakePage:
        def get_text(self) -> str:
            return " Sample text "

    class FakeDocument:
        def __enter__(self):
            return [FakePage()]

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr("app.loaders.pdf_loader.fitz.open", lambda _: FakeDocument())

    pages = load_pdf_pages("data/raw/KPC_2026.05.pdf")

    assert pages == [
        {
            "page_number": 1,
            "text": "Sample text",
            "source_file": "KPC_2026.05.pdf",
        }
    ]