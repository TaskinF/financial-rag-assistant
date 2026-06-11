import pytest

from app.rag.retriever import VectorStoreRetriever


class FakeVectorStore:
    def __init__(self, results: list[dict] | None = None) -> None:
        self.results = results or []
        self.last_query = None
        self.last_top_k = None

    def similarity_search(self, query: str, top_k: int = 3) -> list[dict]:
        self.last_query = query
        self.last_top_k = top_k
        return self.results


def test_retrieve_calls_vector_store_similarity_search_with_query_and_top_k():
    fake_store = FakeVectorStore(results=[])
    retriever = VectorStoreRetriever(vector_store=fake_store)

    retriever.retrieve("revenue growth", top_k=5)

    assert fake_store.last_query == "revenue growth"
    assert fake_store.last_top_k == 5


def test_retrieve_raises_value_error_for_empty_query():
    fake_store = FakeVectorStore()
    retriever = VectorStoreRetriever(vector_store=fake_store)

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_retrieve_raises_value_error_for_none_query():
    fake_store = FakeVectorStore()
    retriever = VectorStoreRetriever(vector_store=fake_store)

    with pytest.raises(ValueError):
        retriever.retrieve(None)


def test_retrieve_raises_value_error_for_non_positive_top_k():
    fake_store = FakeVectorStore()
    retriever = VectorStoreRetriever(vector_store=fake_store)

    with pytest.raises(ValueError):
        retriever.retrieve("revenue growth", top_k=0)


def test_retrieve_returns_similarity_search_results_as_is():
    expected_results = [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "text": "Revenue increased by 12.5%",
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "score": 0.91,
        }
    ]
    fake_store = FakeVectorStore(results=expected_results)
    retriever = VectorStoreRetriever(vector_store=fake_store)

    results = retriever.retrieve("revenue growth", top_k=3)

    assert results == expected_results