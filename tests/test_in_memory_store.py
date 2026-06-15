import pytest

from app.vectorstore.embeddings import FakeEmbeddingModel
from app.vectorstore.in_memory_store import InMemoryVectorStore


def build_store() -> InMemoryVectorStore:
    return InMemoryVectorStore(FakeEmbeddingModel(dimension=8))


def test_cosine_similarity_returns_high_score_for_same_direction_vectors():
    store = build_store()

    score = store._cosine_similarity([1.0, 2.0], [2.0, 4.0])

    assert score == pytest.approx(1.0)


def test_cosine_similarity_returns_zero_for_orthogonal_vectors():
    store = build_store()

    score = store._cosine_similarity([1.0, 0.0], [0.0, 1.0])

    assert score == pytest.approx(0.0)


def test_cosine_similarity_raises_value_error_for_mismatched_dimensions():
    store = build_store()

    with pytest.raises(ValueError):
        store._cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_cosine_similarity_returns_zero_for_zero_vector():
    store = build_store()

    score = store._cosine_similarity([0.0, 0.0], [1.0, 2.0])

    assert score == 0.0


def test_add_documents_does_nothing_for_empty_list():
    store = build_store()

    store.add_documents([])

    assert store._documents == []


def test_add_documents_raises_value_error_when_text_field_is_missing():
    store = build_store()

    with pytest.raises(ValueError):
        store.add_documents(
            [
                {
                    "chunk_id": "sample.pdf_p1_c0",
                    "source_file": "sample.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                }
            ]
        )


def test_add_documents_raises_value_error_for_empty_text():
    store = build_store()

    with pytest.raises(ValueError):
        store.add_documents(
            [
                {
                    "chunk_id": "sample.pdf_p1_c0",
                    "text": "   ",
                    "source_file": "sample.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                }
            ]
        )


def test_add_documents_stores_valid_chunks_with_embeddings():
    store = build_store()
    chunks = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "text": "Revenue increased by 12.5%",
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_index": 0,
        }
    ]

    store.add_documents(chunks)

    assert len(store._documents) == 1
    assert store._documents[0]["chunk_id"] == "sample.pdf_p1_c0"
    assert store._documents[0]["source_file"] == "sample.pdf"
    assert store._documents[0]["page_number"] == 1
    assert store._documents[0]["chunk_index"] == 0
    assert store._documents[0]["text"] == "Revenue increased by 12.5%"
    assert "embedding" in store._documents[0]


def test_add_embedded_documents_does_nothing_for_empty_list():
    store = build_store()

    store.add_embedded_documents([])

    assert store._documents == []


def test_add_embedded_documents_stores_valid_documents():
    store = build_store()
    documents = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "text": "Revenue increased by 12.5%",
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "embedding": [0.1, 0.2, 0.3],
        }
    ]

    store.add_embedded_documents(documents)

    assert len(store._documents) == 1
    assert store._documents[0]["chunk_id"] == "sample.pdf_p1_c0"
    assert store._documents[0]["text"] == "Revenue increased by 12.5%"
    assert store._documents[0]["embedding"] == [0.1, 0.2, 0.3]


def test_add_embedded_documents_raises_value_error_when_embedding_is_missing():
    store = build_store()

    with pytest.raises(ValueError):
        store.add_embedded_documents(
            [
                {
                    "chunk_id": "sample.pdf_p1_c0",
                    "text": "Revenue increased by 12.5%",
                    "source_file": "sample.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                }
            ]
        )


def test_add_embedded_documents_raises_value_error_for_empty_embedding():
    store = build_store()

    with pytest.raises(ValueError):
        store.add_embedded_documents(
            [
                {
                    "chunk_id": "sample.pdf_p1_c0",
                    "text": "Revenue increased by 12.5%",
                    "source_file": "sample.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                    "embedding": [],
                }
            ]
        )


def test_add_embedded_documents_raises_value_error_when_text_is_missing():
    store = build_store()

    with pytest.raises(ValueError):
        store.add_embedded_documents(
            [
                {
                    "chunk_id": "sample.pdf_p1_c0",
                    "source_file": "sample.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                }
            ]
        )


def test_add_embedded_documents_raises_value_error_for_empty_text():
    store = build_store()

    with pytest.raises(ValueError):
        store.add_embedded_documents(
            [
                {
                    "chunk_id": "sample.pdf_p1_c0",
                    "text": "   ",
                    "source_file": "sample.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                }
            ]
        )


def test_get_documents_returns_documents_with_embeddings():
    store = build_store()
    documents = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "text": "Revenue increased by 12.5%",
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "embedding": [0.1, 0.2, 0.3],
        }
    ]

    store.add_embedded_documents(documents)
    stored_documents = store.get_documents()

    assert len(stored_documents) == 1
    assert stored_documents[0]["chunk_id"] == "sample.pdf_p1_c0"
    assert stored_documents[0]["embedding"] == [0.1, 0.2, 0.3]


def test_similarity_search_returns_empty_list_when_store_is_empty():
    store = build_store()

    results = store.similarity_search("revenue growth")

    assert results == []


def test_similarity_search_raises_value_error_for_empty_query():
    store = build_store()

    with pytest.raises(ValueError):
        store.similarity_search("")


def test_similarity_search_raises_value_error_for_none_query():
    store = build_store()

    with pytest.raises(ValueError):
        store.similarity_search(None)


def test_similarity_search_raises_value_error_for_non_positive_top_k():
    store = build_store()

    with pytest.raises(ValueError):
        store.similarity_search("revenue", top_k=0)


def test_similarity_search_returns_scored_results_without_embedding_field():
    store = build_store()
    chunks = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "text": "Revenue increased significantly in 2024.",
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_index": 0,
        },
        {
            "chunk_id": "sample.pdf_p2_c0",
            "text": "Operating expenses decreased in the second quarter.",
            "source_file": "sample.pdf",
            "page_number": 2,
            "chunk_index": 0,
        },
    ]
    store.add_documents(chunks)

    results = store.similarity_search("revenue increased", top_k=2)

    assert len(results) == 2
    assert "score" in results[0]
    assert "embedding" not in results[0]


def test_similarity_search_returns_results_sorted_by_score_descending():
    store = build_store()
    chunks = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "text": "Revenue increased significantly in 2024.",
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_index": 0,
        },
        {
            "chunk_id": "sample.pdf_p2_c0",
            "text": "Operating expenses decreased in the second quarter.",
            "source_file": "sample.pdf",
            "page_number": 2,
            "chunk_index": 0,
        },
        {
            "chunk_id": "sample.pdf_p3_c0",
            "text": "Cash flow remained stable throughout the year.",
            "source_file": "sample.pdf",
            "page_number": 3,
            "chunk_index": 0,
        },
    ]
    store.add_documents(chunks)

    results = store.similarity_search("revenue increased", top_k=3)

    assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]


def test_similarity_search_does_not_return_more_than_top_k_results():
    store = build_store()
    chunks = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "text": "Revenue increased significantly in 2024.",
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_index": 0,
        },
        {
            "chunk_id": "sample.pdf_p2_c0",
            "text": "Operating expenses decreased in the second quarter.",
            "source_file": "sample.pdf",
            "page_number": 2,
            "chunk_index": 0,
        },
        {
            "chunk_id": "sample.pdf_p3_c0",
            "text": "Cash flow remained stable throughout the year.",
            "source_file": "sample.pdf",
            "page_number": 3,
            "chunk_index": 0,
        },
    ]
    store.add_documents(chunks)

    results = store.similarity_search("financial performance", top_k=2)

    assert len(results) == 2


def test_similarity_search_preserves_chunk_metadata():
    store = build_store()
    chunks = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "text": "Revenue increased significantly in 2024.",
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_index": 0,
        }
    ]
    store.add_documents(chunks)

    results = store.similarity_search("revenue", top_k=1)

    assert results[0]["chunk_id"] == "sample.pdf_p1_c0"
    assert results[0]["source_file"] == "sample.pdf"
    assert results[0]["page_number"] == 1
    assert results[0]["chunk_index"] == 0