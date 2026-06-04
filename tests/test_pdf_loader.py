import pytest

from app.loaders.pdf_loader import load_pdf_pages


def test_load_pdf_pages_raises_file_not_found_for_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pdf_pages("missing_file.pdf")


def test_load_pdf_pages_raises_value_error_for_non_pdf_file():
    with pytest.raises(ValueError):
        load_pdf_pages("README.md")