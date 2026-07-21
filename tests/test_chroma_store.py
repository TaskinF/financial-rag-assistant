import pytest

pytest.importorskip("chromadb")

from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.embeddings import FakeEmbeddingModel


def build_store(tmp_path) -> ChromaVectorStore:
    return ChromaVectorStore(
        embedding_model=FakeEmbeddingModel(dimension=8),
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_financial_documents",
    )


def build_chunk(
    chunk_id: str,
    text: str,
    source_file: str,
    page_number: int,
    chunk_index: int,
    document_id: str | None = None,
) -> dict:
    chunk = {
        "chunk_id": chunk_id,
        "text": text,
        "source_file": source_file,
        "page_number": page_number,
        "chunk_index": chunk_index,
    }

    if document_id is not None:
        chunk["document_id"] = document_id

    return chunk


def build_multi_document_chunks() -> list[dict]:
    return [
        build_chunk(
            chunk_id="doc_a_p1_c0",
            text="Revenue increased strongly for document A.",
            source_file="doc_a.pdf",
            page_number=1,
            chunk_index=0,
            document_id="doc_a",
        ),
        build_chunk(
            chunk_id="doc_a_p2_c0",
            text="Net income improved in document A.",
            source_file="doc_a.pdf",
            page_number=2,
            chunk_index=0,
            document_id="doc_a",
        ),
        build_chunk(
            chunk_id="doc_b_p1_c0",
            text="Cash flow remained stable for document B.",
            source_file="doc_b.pdf",
            page_number=1,
            chunk_index=0,
            document_id="doc_b",
        ),
        build_chunk(
            chunk_id="doc_b_p2_c0",
            text="Operating margin expanded in document B.",
            source_file="doc_b.pdf",
            page_number=2,
            chunk_index=0,
            document_id="doc_b",
        ),
    ]


def test_constructor_creates_store_with_valid_arguments(tmp_path):
    store = build_store(tmp_path)

    assert store.embedding_model is not None
    assert store.persist_dir.exists()
    assert store.count() == 0


def test_constructor_raises_value_error_when_embedding_model_is_none(tmp_path):
    with pytest.raises(ValueError):
        ChromaVectorStore(
            embedding_model=None,
            persist_dir=str(tmp_path / "chroma"),
        )


def test_add_documents_does_nothing_for_empty_list(tmp_path):
    store = build_store(tmp_path)

    store.add_documents([])

    assert store.count() == 0


def test_add_documents_adds_valid_chunk_and_updates_count(tmp_path):
    store = build_store(tmp_path)
    chunks = [
        build_chunk(
            chunk_id="sample.pdf_p1_c0",
            text="Revenue increased by 12.5% in the first quarter.",
            source_file="sample.pdf",
            page_number=1,
            chunk_index=0,
        )
    ]

    store.add_documents(chunks)

    assert store.count() == 1


def test_add_documents_upsert_does_not_create_duplicate_for_same_chunk_id(tmp_path):
    store = build_store(tmp_path)
    chunks = [
        build_chunk(
            chunk_id="sample.pdf_p1_c0",
            text="Revenue increased by 12.5% in the first quarter.",
            source_file="sample.pdf",
            page_number=1,
            chunk_index=0,
        )
    ]

    store.add_documents(chunks)
    store.add_documents(chunks)

    assert store.count() == 1


def test_similarity_search_returns_valid_result_format(tmp_path):
    store = build_store(tmp_path)
    store.add_documents(
        [
            build_chunk(
                chunk_id="sample.pdf_p1_c0",
                text="Revenue increased by 12.5% in the first quarter.",
                source_file="sample.pdf",
                page_number=1,
                chunk_index=0,
            )
        ]
    )

    results = store.similarity_search("revenue increased", top_k=1)

    assert len(results) == 1
    assert "text" in results[0]
    assert "document_id" in results[0]
    assert "source_file" in results[0]
    assert "page_number" in results[0]
    assert "chunk_id" in results[0]
    assert "score" in results[0]
    assert results[0]["chunk_id"] == "sample.pdf_p1_c0"


def test_similarity_search_raises_value_error_for_empty_query(tmp_path):
    store = build_store(tmp_path)

    with pytest.raises(ValueError):
        store.similarity_search("")


def test_similarity_search_raises_value_error_for_non_positive_top_k(tmp_path):
    store = build_store(tmp_path)

    with pytest.raises(ValueError):
        store.similarity_search("revenue", top_k=0)


def test_add_embedded_documents_accepts_valid_precomputed_embeddings(tmp_path):
    store = build_store(tmp_path)
    embedding_model = FakeEmbeddingModel(dimension=8)
    text = "Net income improved compared with last year."

    store.add_embedded_documents(
        [
            {
                **build_chunk(
                    chunk_id="sample.pdf_p2_c0",
                    text=text,
                    source_file="sample.pdf",
                    page_number=2,
                    chunk_index=0,
                    document_id="doc_embedded",
                ),
                "embedding": embedding_model.embed_text(text),
            }
        ]
    )

    assert store.count() == 1
    results = store.similarity_search(
        "net income improved",
        top_k=1,
        metadata_filter={"document_id": "doc_embedded"},
    )

    assert len(results) == 1
    assert results[0]["document_id"] == "doc_embedded"


def test_add_embedded_documents_raises_value_error_when_embedding_is_missing(tmp_path):
    store = build_store(tmp_path)

    with pytest.raises(ValueError):
        store.add_embedded_documents(
            [
                {
                    **build_chunk(
                        chunk_id="sample.pdf_p2_c0",
                        text="Net income improved compared with last year.",
                        source_file="sample.pdf",
                        page_number=2,
                        chunk_index=0,
                        document_id="doc_embedded",
                    ),
                }
            ]
        )


def test_clear_removes_all_documents(tmp_path):
    store = build_store(tmp_path)
    store.add_documents(
        [
            build_chunk(
                chunk_id="sample.pdf_p1_c0",
                text="Revenue increased by 12.5% in the first quarter.",
                source_file="sample.pdf",
                page_number=1,
                chunk_index=0,
            )
        ]
    )

    store.clear()

    assert store.count() == 0


def test_similarity_search_filters_results_by_document_id_doc_a(tmp_path):
    store = build_store(tmp_path)
    store.add_documents(build_multi_document_chunks())

    results = store.similarity_search(
        "document revenue income",
        top_k=5,
        metadata_filter={"document_id": "doc_a"},
    )

    assert results
    assert all(result["document_id"] == "doc_a" for result in results)


def test_similarity_search_filters_results_by_document_id_doc_b(tmp_path):
    store = build_store(tmp_path)
    store.add_documents(build_multi_document_chunks())

    results = store.similarity_search(
        "document cash margin",
        top_k=5,
        metadata_filter={"document_id": "doc_b"},
    )

    assert results
    assert all(result["document_id"] == "doc_b" for result in results)


def test_similarity_search_returns_empty_list_for_missing_document_id_filter(tmp_path):
    store = build_store(tmp_path)
    store.add_documents(build_multi_document_chunks())

    results = store.similarity_search(
        "financial performance",
        top_k=5,
        metadata_filter={"document_id": "missing_doc"},
    )

    assert results == []


def test_similarity_search_without_metadata_filter_preserves_existing_behavior(tmp_path):
    store = build_store(tmp_path)
    store.add_documents(build_multi_document_chunks())

    results = store.similarity_search("financial performance", top_k=3)

    assert len(results) <= 3
    assert all("text" in result for result in results)


def test_delete_by_document_id_removes_only_matching_documents(tmp_path):
    store = build_store(tmp_path)
    store.add_documents(build_multi_document_chunks())

    store.delete_by_document_id("doc_a")

    doc_a_results = store.similarity_search(
        "financial performance",
        top_k=5,
        metadata_filter={"document_id": "doc_a"},
    )
    doc_b_results = store.similarity_search(
        "financial performance",
        top_k=5,
        metadata_filter={"document_id": "doc_b"},
    )

    assert doc_a_results == []
    assert doc_b_results
    assert all(result["document_id"] == "doc_b" for result in doc_b_results)


def test_delete_by_document_id_rejects_empty_document_id(tmp_path):
    store = build_store(tmp_path)

    with pytest.raises(ValueError):
        store.delete_by_document_id("   ")
