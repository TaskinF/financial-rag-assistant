import app.processing.ingestion as ingestion


def test_build_chunks_from_pdf_cleans_text_and_preserves_metadata(monkeypatch):
    fake_pages = [
        {
            "text": "  Revenue\tGrowth\n\n 12.5% ",
            "source_file": "sample.pdf",
            "page_number": 1,
        }
    ]

    def fake_load_pdf_pages(file_path: str) -> list[dict]:
        assert file_path == "sample.pdf"
        return fake_pages

    def fake_clean_text(text: str | None) -> str:
        assert text == "  Revenue\tGrowth\n\n 12.5% "
        return "Revenue Growth\n12.5%"

    def fake_chunk_pages(
        pages: list[dict],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[dict]:
        assert pages == [
            {
                "text": "Revenue Growth\n12.5%",
                "source_file": "sample.pdf",
                "page_number": 1,
            }
        ]
        assert chunk_size == 1000
        assert chunk_overlap == 200

        return [
            {
                "chunk_id": "sample.pdf_p1_c0",
                "text": "Revenue Growth\n12.5%",
                "source_file": "sample.pdf",
                "page_number": 1,
                "chunk_index": 0,
            }
        ]

    monkeypatch.setattr(ingestion, "load_pdf_pages", fake_load_pdf_pages)
    monkeypatch.setattr(ingestion, "clean_text", fake_clean_text)
    monkeypatch.setattr(ingestion, "chunk_pages", fake_chunk_pages)

    chunks = ingestion.build_chunks_from_pdf("sample.pdf")

    assert chunks == [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "text": "Revenue Growth\n12.5%",
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_index": 0,
        }
    ]


def test_build_chunks_from_pdf_returns_empty_list_when_loader_returns_no_pages(monkeypatch):
    def fake_load_pdf_pages(file_path: str) -> list[dict]:
        assert file_path == "empty.pdf"
        return []

    def fake_clean_text(text: str | None) -> str:
        raise AssertionError("clean_text should not be called for empty page lists")

    def fake_chunk_pages(
        pages: list[dict],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[dict]:
        assert pages == []
        return []

    monkeypatch.setattr(ingestion, "load_pdf_pages", fake_load_pdf_pages)
    monkeypatch.setattr(ingestion, "clean_text", fake_clean_text)
    monkeypatch.setattr(ingestion, "chunk_pages", fake_chunk_pages)

    chunks = ingestion.build_chunks_from_pdf("empty.pdf")

    assert chunks == []