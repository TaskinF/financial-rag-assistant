from pathlib import Path

import pytest

from app.services.document_indexing_service import DocumentIndexingService


class FakeVectorStore:
    def __init__(self) -> None:
        self.added_chunks: list[dict] = []

    def add_documents(self, chunks: list[dict]) -> None:
        self.added_chunks = [dict(chunk) for chunk in chunks]


def build_service(tmp_path) -> DocumentIndexingService:
    return DocumentIndexingService(
        documents_dir=str(tmp_path / "documents"),
        registry_path=str(tmp_path / "document_registry.json"),
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="financial_documents",
    )


def create_fake_pdf(tmp_path, name: str = "sample.pdf") -> Path:
    pdf_path = tmp_path / name
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")
    return pdf_path


def sample_chunks() -> list[dict]:
    return [
        {
            "text": "Fon Yonetim Ucreti 7.125.856,54",
            "source_file": "sample.pdf",
            "page_number": 3,
            "chunk_index": 0,
            "chunk_id": "sample.pdf_p3_c0",
        }
    ]


def test_index_pdf_raises_file_not_found_error_for_missing_pdf(tmp_path):
    service = build_service(tmp_path)

    with pytest.raises(FileNotFoundError):
        service.index_pdf(str(tmp_path / "missing.pdf"))


def test_index_pdf_raises_value_error_for_non_pdf_file(tmp_path):
    service = build_service(tmp_path)
    text_file = tmp_path / "sample.txt"
    text_file.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(ValueError):
        service.index_pdf(str(text_file))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_overlap": -1},
        {"chunk_size": 200, "chunk_overlap": 200},
    ],
)
def test_index_pdf_raises_value_error_for_invalid_chunk_settings(tmp_path, kwargs):
    service = build_service(tmp_path)
    pdf_path = create_fake_pdf(tmp_path)

    with pytest.raises(ValueError):
        service.index_pdf(str(pdf_path), **kwargs)


def test_index_pdf_generates_document_id_when_not_provided(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    pdf_path = create_fake_pdf(tmp_path, name="KPC_2026.05.pdf")
    fake_store = FakeVectorStore()

    monkeypatch.setattr(
        "app.services.document_indexing_service.build_chunks_from_pdf",
        lambda *args, **kwargs: sample_chunks(),
    )
    monkeypatch.setattr(service, "_get_vector_store", lambda: fake_store)

    result = service.index_pdf(str(pdf_path))

    assert result["document_id"]
    assert result["filename"] == f"{result['document_id']}.pdf"


def test_index_pdf_copies_pdf_into_documents_dir(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    pdf_path = create_fake_pdf(tmp_path)
    fake_store = FakeVectorStore()

    monkeypatch.setattr(
        "app.services.document_indexing_service.build_chunks_from_pdf",
        lambda *args, **kwargs: sample_chunks(),
    )
    monkeypatch.setattr(service, "_get_vector_store", lambda: fake_store)

    result = service.index_pdf(str(pdf_path), document_id="doc_a")
    saved_path = Path(result["path"])

    assert saved_path.exists()
    assert saved_path.parent == Path(service.documents_dir)


def test_index_pdf_adds_document_id_to_chunks(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    pdf_path = create_fake_pdf(tmp_path)
    fake_store = FakeVectorStore()

    monkeypatch.setattr(
        "app.services.document_indexing_service.build_chunks_from_pdf",
        lambda *args, **kwargs: sample_chunks(),
    )
    monkeypatch.setattr(service, "_get_vector_store", lambda: fake_store)

    service.index_pdf(str(pdf_path), document_id="doc_a")

    assert fake_store.added_chunks[0]["document_id"] == "doc_a"


def test_index_pdf_normalizes_source_file_to_saved_pdf_name(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    pdf_path = create_fake_pdf(tmp_path)
    fake_store = FakeVectorStore()

    monkeypatch.setattr(
        "app.services.document_indexing_service.build_chunks_from_pdf",
        lambda *args, **kwargs: sample_chunks(),
    )
    monkeypatch.setattr(service, "_get_vector_store", lambda: fake_store)

    service.index_pdf(str(pdf_path), document_id="doc_a")

    assert fake_store.added_chunks[0]["source_file"] == "doc_a.pdf"


def test_index_pdf_calls_add_documents_with_expected_chunks(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    pdf_path = create_fake_pdf(tmp_path)
    fake_store = FakeVectorStore()
    chunks = sample_chunks()

    monkeypatch.setattr(
        "app.services.document_indexing_service.build_chunks_from_pdf",
        lambda *args, **kwargs: chunks,
    )
    monkeypatch.setattr(service, "_get_vector_store", lambda: fake_store)

    service.index_pdf(str(pdf_path), document_id="doc_a")

    assert len(fake_store.added_chunks) == 1
    assert fake_store.added_chunks[0]["chunk_id"] == "sample.pdf_p3_c0"


def test_index_pdf_creates_registry_record_with_required_fields(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    pdf_path = create_fake_pdf(tmp_path)
    fake_store = FakeVectorStore()

    monkeypatch.setattr(
        "app.services.document_indexing_service.build_chunks_from_pdf",
        lambda *args, **kwargs: sample_chunks(),
    )
    monkeypatch.setattr(service, "_get_vector_store", lambda: fake_store)

    result = service.index_pdf(str(pdf_path), document_id="doc_a")

    assert result["document_id"] == "doc_a"
    assert result["filename"] == "doc_a.pdf"
    assert result["chunk_count"] == 1
    assert result["collection_name"] == "financial_documents"
    assert result["status"] == "indexed"
    assert "indexed_at" in result


def test_index_pdf_start_page_and_end_page_filter_work(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    pdf_path = create_fake_pdf(tmp_path)
    fake_store = FakeVectorStore()
    chunks = [
        {
            "text": "Page 2 chunk",
            "source_file": "sample.pdf",
            "page_number": 2,
            "chunk_index": 0,
            "chunk_id": "sample.pdf_p2_c0",
        },
        {
            "text": "Page 3 chunk",
            "source_file": "sample.pdf",
            "page_number": 3,
            "chunk_index": 0,
            "chunk_id": "sample.pdf_p3_c0",
        },
        {
            "text": "Page 4 chunk",
            "source_file": "sample.pdf",
            "page_number": 4,
            "chunk_index": 0,
            "chunk_id": "sample.pdf_p4_c0",
        },
    ]

    monkeypatch.setattr(
        "app.services.document_indexing_service.build_chunks_from_pdf",
        lambda *args, **kwargs: chunks,
    )
    monkeypatch.setattr(service, "_get_vector_store", lambda: fake_store)

    result = service.index_pdf(
        str(pdf_path),
        document_id="doc_a",
        start_page=3,
        end_page=3,
    )

    assert result["chunk_count"] == 1
    assert fake_store.added_chunks[0]["page_number"] == 3


def test_index_pdf_raises_value_error_when_no_chunks_remain_after_filtering(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    pdf_path = create_fake_pdf(tmp_path)
    fake_store = FakeVectorStore()

    monkeypatch.setattr(
        "app.services.document_indexing_service.build_chunks_from_pdf",
        lambda *args, **kwargs: sample_chunks(),
    )
    monkeypatch.setattr(service, "_get_vector_store", lambda: fake_store)

    with pytest.raises(ValueError):
        service.index_pdf(
            str(pdf_path),
            document_id="doc_a",
            start_page=10,
            end_page=12,
        )


def test_list_documents_returns_registry_records(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    pdf_path = create_fake_pdf(tmp_path)
    fake_store = FakeVectorStore()

    monkeypatch.setattr(
        "app.services.document_indexing_service.build_chunks_from_pdf",
        lambda *args, **kwargs: sample_chunks(),
    )
    monkeypatch.setattr(service, "_get_vector_store", lambda: fake_store)

    service.index_pdf(str(pdf_path), document_id="doc_a")

    documents = service.list_documents()

    assert len(documents) == 1
    assert documents[0]["document_id"] == "doc_a"


def test_get_document_returns_expected_record(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    pdf_path = create_fake_pdf(tmp_path)
    fake_store = FakeVectorStore()

    monkeypatch.setattr(
        "app.services.document_indexing_service.build_chunks_from_pdf",
        lambda *args, **kwargs: sample_chunks(),
    )
    monkeypatch.setattr(service, "_get_vector_store", lambda: fake_store)

    service.index_pdf(str(pdf_path), document_id="doc_a")

    document = service.get_document("doc_a")

    assert document is not None
    assert document["document_id"] == "doc_a"
