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
        {
            "chunk_id": "sample.pdf_p1_c0",
            "text": "Revenue increased by 12.5% in the first quarter.",
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_index": 0,
        }
    ]

    store.add_documents(chunks)

    assert store.count() == 1


def test_add_documents_upsert_does_not_create_duplicate_for_same_chunk_id(tmp_path):
    store = build_store(tmp_path)
    chunks = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "text": "Revenue increased by 12.5% in the first quarter.",
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_index": 0,
        }
    ]

    store.add_documents(chunks)
    store.add_documents(chunks)

    assert store.count() == 1


def test_similarity_search_returns_valid_result_format(tmp_path):
    store = build_store(tmp_path)
    store.add_documents(
        [
            {
                "chunk_id": "sample.pdf_p1_c0",
                "text": "Revenue increased by 12.5% in the first quarter.",
                "source_file": "sample.pdf",
                "page_number": 1,
                "chunk_index": 0,
            }
        ]
    )

    results = store.similarity_search("revenue increased", top_k=1)

    assert len(results) == 1
    assert "text" in results[0]
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
                "chunk_id": "sample.pdf_p2_c0",
                "text": text,
                "source_file": "sample.pdf",
                "page_number": 2,
                "chunk_index": 0,
                "embedding": embedding_model.embed_text(text),
            }
        ]
    )

    assert store.count() == 1


def test_add_embedded_documents_raises_value_error_when_embedding_is_missing(tmp_path):
    store = build_store(tmp_path)

    with pytest.raises(ValueError):
        store.add_embedded_documents(
            [
                {
                    "chunk_id": "sample.pdf_p2_c0",
                    "text": "Net income improved compared with last year.",
                    "source_file": "sample.pdf",
                    "page_number": 2,
                    "chunk_index": 0,
                }
            ]
        )


def test_clear_removes_all_documents(tmp_path):
    store = build_store(tmp_path)
    store.add_documents(
        [
            {
                "chunk_id": "sample.pdf_p1_c0",
                "text": "Revenue increased by 12.5% in the first quarter.",
                "source_file": "sample.pdf",
                "page_number": 1,
                "chunk_index": 0,
            }
        ]
    )

    store.clear()

    assert store.count() == 0
