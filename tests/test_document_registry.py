import json
import re

import pytest

from app.documents.document_registry import DocumentRegistry, create_document_id


def build_registry(tmp_path) -> DocumentRegistry:
    return DocumentRegistry(registry_path=str(tmp_path / "document_registry.json"))


def sample_document(document_id: str = "kpc_2026_05_b02bb78b") -> dict:
    return {
        "document_id": document_id,
        "filename": "KPC_2026.05.pdf",
        "path": "data/documents/KPC_2026.05.pdf",
        "chunk_count": 42,
        "collection_name": "financial_documents",
        "indexed_at": "2026-07-19T10:00:00Z",
        "status": "indexed",
    }


def test_registry_file_is_created_automatically_when_missing(tmp_path):
    registry_path = tmp_path / "document_registry.json"

    assert not registry_path.exists()

    build_registry(tmp_path)

    assert registry_path.exists()


def test_registry_starts_with_empty_documents_list(tmp_path):
    registry_path = tmp_path / "document_registry.json"
    build_registry(tmp_path)

    with registry_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data == {"documents": []}


def test_list_documents_returns_empty_list_initially(tmp_path):
    registry = build_registry(tmp_path)

    assert registry.list_documents() == []


def test_register_document_adds_new_record(tmp_path):
    registry = build_registry(tmp_path)
    document = sample_document()

    saved_document = registry.register_document(document)

    assert saved_document == document
    assert registry.list_documents() == [document]


def test_register_document_updates_existing_record_without_creating_duplicate(tmp_path):
    registry = build_registry(tmp_path)
    document = sample_document()
    updated_document = sample_document()
    updated_document["chunk_count"] = 100
    updated_document["status"] = "reindexed"

    registry.register_document(document)
    registry.register_document(updated_document)

    documents = registry.list_documents()

    assert len(documents) == 1
    assert documents[0]["chunk_count"] == 100
    assert documents[0]["status"] == "reindexed"


def test_get_document_returns_existing_record(tmp_path):
    registry = build_registry(tmp_path)
    document = sample_document()
    registry.register_document(document)

    result = registry.get_document(document["document_id"])

    assert result == document


def test_get_document_returns_none_for_missing_document_id(tmp_path):
    registry = build_registry(tmp_path)

    assert registry.get_document("missing_document") is None


def test_delete_document_removes_existing_record_and_returns_true(tmp_path):
    registry = build_registry(tmp_path)
    document = sample_document()
    registry.register_document(document)

    deleted = registry.delete_document(document["document_id"])

    assert deleted is True
    assert registry.list_documents() == []


def test_delete_document_returns_false_for_missing_record(tmp_path):
    registry = build_registry(tmp_path)

    assert registry.delete_document("missing_document") is False


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("get_document", ("",)),
        ("delete_document", ("   ",)),
    ],
)
def test_methods_raise_value_error_for_empty_document_id(tmp_path, method_name, args):
    registry = build_registry(tmp_path)

    with pytest.raises(ValueError):
        getattr(registry, method_name)(*args)


def test_register_document_raises_value_error_when_document_id_is_missing(tmp_path):
    registry = build_registry(tmp_path)
    document = sample_document()
    document.pop("document_id")

    with pytest.raises(ValueError):
        registry.register_document(document)


def test_create_document_id_is_safe_and_deterministic_for_known_filename():
    document_id_1 = create_document_id("KPC_2026.05.pdf")
    document_id_2 = create_document_id("KPC_2026.05.pdf")

    assert document_id_1 == document_id_2
    assert document_id_1.startswith("kpc_2026_05_")
    assert re.fullmatch(r"[a-z0-9_]+", document_id_1) is not None


def test_create_document_id_normalizes_turkish_characters():
    document_id = create_document_id("\u00c7a\u011fr\u0131_\u00d6\u011frenci_\u015e\u00fcbesi.pdf")

    assert document_id.startswith("cagri_ogrenci_subesi_")
    assert re.fullmatch(r"[a-z0-9_]+", document_id) is not None


def test_create_document_id_raises_value_error_for_empty_filename():
    with pytest.raises(ValueError):
        create_document_id("   ")
