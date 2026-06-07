import pytest

from app.vectorstore.embeddings import BGEFlagEmbeddingModel, FakeEmbeddingModel


def test_fake_embedding_model_returns_same_embedding_for_same_text():
    model = FakeEmbeddingModel(dimension=8)

    embedding_1 = model.embed_text("Revenue increased by 12.5%")
    embedding_2 = model.embed_text("Revenue increased by 12.5%")

    assert embedding_1 == embedding_2


def test_fake_embedding_model_returns_different_embeddings_for_different_texts():
    model = FakeEmbeddingModel(dimension=8)

    embedding_1 = model.embed_text("Revenue increased by 12.5%")
    embedding_2 = model.embed_text("Net income decreased by 3.2%")

    assert embedding_1 != embedding_2


def test_fake_embedding_model_respects_dimension_parameter():
    model = FakeEmbeddingModel(dimension=16)

    embedding = model.embed_text("Financial report")

    assert len(embedding) == 16


def test_fake_embedding_model_raises_value_error_for_non_positive_dimension():
    with pytest.raises(ValueError):
        FakeEmbeddingModel(dimension=0)


def test_fake_embedding_model_raises_value_error_for_empty_text():
    model = FakeEmbeddingModel()

    with pytest.raises(ValueError):
        model.embed_text("")


def test_fake_embedding_model_raises_value_error_for_none_text():
    model = FakeEmbeddingModel()

    with pytest.raises(ValueError):
        model.embed_text(None)


def test_fake_embedding_model_embed_documents_returns_list_of_embeddings():
    model = FakeEmbeddingModel(dimension=8)

    embeddings = model.embed_documents(
        ["Revenue growth", "Net income", "Cash flow"]
    )

    assert isinstance(embeddings, list)
    assert all(isinstance(embedding, list) for embedding in embeddings)


def test_fake_embedding_model_embed_documents_returns_same_count_as_input():
    model = FakeEmbeddingModel(dimension=8)
    texts = ["Revenue growth", "Net income", "Cash flow"]

    embeddings = model.embed_documents(texts)

    assert len(embeddings) == len(texts)


def test_bge_flag_embedding_model_class_exists():
    assert BGEFlagEmbeddingModel is not None