from app.vectorstore.embedding_cache import (
    build_embedding_cache_key,
    load_embedding_cache,
    save_embedding_cache,
)


def sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "sample.pdf_p1_c0",
            "text": "Revenue increased by 12.5%",
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_index": 0,
        },
        {
            "chunk_id": "sample.pdf_p2_c0",
            "text": "Net income improved in Q2",
            "source_file": "sample.pdf",
            "page_number": 2,
            "chunk_index": 0,
        },
    ]


def test_build_embedding_cache_key_is_deterministic_for_same_input():
    chunks = sample_chunks()

    key_1 = build_embedding_cache_key(chunks, model_name="BAAI/bge-m3")
    key_2 = build_embedding_cache_key(chunks, model_name="BAAI/bge-m3")

    assert key_1 == key_2


def test_build_embedding_cache_key_changes_when_text_changes():
    chunks = sample_chunks()
    modified_chunks = sample_chunks()
    modified_chunks[0]["text"] = "Revenue decreased by 3.2%"

    key_1 = build_embedding_cache_key(chunks, model_name="BAAI/bge-m3")
    key_2 = build_embedding_cache_key(modified_chunks, model_name="BAAI/bge-m3")

    assert key_1 != key_2


def test_build_embedding_cache_key_changes_when_model_name_changes():
    chunks = sample_chunks()

    key_1 = build_embedding_cache_key(chunks, model_name="BAAI/bge-m3")
    key_2 = build_embedding_cache_key(chunks, model_name="another-model")

    assert key_1 != key_2


def test_save_and_load_embedding_cache_roundtrip(tmp_path):
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
    cache_key = "test_cache_key"

    saved_path = save_embedding_cache(tmp_path, cache_key, documents)
    loaded_documents = load_embedding_cache(tmp_path, cache_key)

    assert saved_path.exists()
    assert loaded_documents == documents


def test_load_embedding_cache_returns_none_when_file_does_not_exist(tmp_path):
    loaded_documents = load_embedding_cache(tmp_path, "missing_cache_key")

    assert loaded_documents is None