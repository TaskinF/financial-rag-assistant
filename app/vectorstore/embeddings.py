from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod


def _validate_text(text: str | None) -> str:
    """
    Validate a single text input for embedding.

    Args:
        text: Input text to validate.

    Returns:
        Normalized non-empty text.

    Raises:
        ValueError: If text is None or empty.
    """
    if text is None:
        raise ValueError("text cannot be None")

    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("text cannot be empty")

    return normalized_text


class EmbeddingModel(ABC):
    """
    Base interface for embedding providers.
    """

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """
        Generate a dense embedding vector for a single text.

        Args:
            text: Input text.

        Returns:
            Dense embedding vector as a list of floats.
        """
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate dense embedding vectors for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of dense embedding vectors.
        """
        return [self.embed_text(text) for text in texts]


class BGEFlagEmbeddingModel(EmbeddingModel):
    """
    Dense embedding model backed by FlagEmbedding BGEM3FlagModel.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
    ) -> None:
        """
        Initialize the BGE M3 embedding model.

        Args:
            model_name: Model name to load.
            use_fp16: Whether to load the model in fp16 mode.
            return_sparse: Whether the model should also compute sparse outputs.
            return_colbert_vecs: Whether the model should also compute ColBERT vectors.
        """
        from FlagEmbedding import BGEM3FlagModel

        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.return_sparse = return_sparse
        self.return_colbert_vecs = return_colbert_vecs
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def embed_text(self, text: str) -> list[float]:
        """
        Generate a dense embedding vector for a single text.

        Args:
            text: Input text.

        Returns:
            Dense embedding vector as a list of floats.

        Raises:
            ValueError: If text is None or empty.
        """
        normalized_text = _validate_text(text)

        output = self.model.encode(
            [normalized_text],
            return_dense=True,
            return_sparse=self.return_sparse,
            return_colbert_vecs=self.return_colbert_vecs,
        )

        return output["dense_vecs"][0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate dense embedding vectors for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            List of dense embedding vectors.

        Raises:
            ValueError: If any text is None or empty.
        """
        if not texts:
            return []

        normalized_texts = [_validate_text(text) for text in texts]

        output = self.model.encode(
            normalized_texts,
            return_dense=True,
            return_sparse=self.return_sparse,
            return_colbert_vecs=self.return_colbert_vecs,
        )

        return [embedding.tolist() for embedding in output["dense_vecs"]]


class FakeEmbeddingModel(EmbeddingModel):
    """
    Deterministic hash-based embedding model for tests.
    """

    def __init__(self, dimension: int = 8) -> None:
        """
        Initialize the fake embedding model.

        Args:
            dimension: Output embedding size.

        Raises:
            ValueError: If dimension is not positive.
        """
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")

        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        """
        Generate a deterministic embedding vector for a single text.

        Args:
            text: Input text.

        Returns:
            Deterministic embedding vector as a list of floats.

        Raises:
            ValueError: If text is None or empty.
        """
        normalized_text = _validate_text(text)
        values: list[float] = []
        counter = 0

        while len(values) < self.dimension:
            payload = f"{normalized_text}:{counter}".encode("utf-8")
            digest = hashlib.sha256(payload).digest()

            for byte in digest:
                values.append(byte / 255.0)
                if len(values) == self.dimension:
                    break

            counter += 1

        return values