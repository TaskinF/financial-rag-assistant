import hashlib
import pickle
from pathlib import Path


def build_embedding_cache_key(
    chunks: list[dict],
    model_name: str,
    cache_version: str = "v1",
) -> str:
    """
    Build a deterministic cache key for a chunk list and embedding model.
    """
    hasher = hashlib.sha256()
    hasher.update(cache_version.encode("utf-8"))
    hasher.update(model_name.encode("utf-8"))

    for chunk in chunks:
        hasher.update(str(chunk.get("chunk_id", "")).encode("utf-8"))
        hasher.update(str(chunk.get("source_file", "")).encode("utf-8"))
        hasher.update(str(chunk.get("page_number", "")).encode("utf-8"))
        hasher.update(str(chunk.get("text", "")).encode("utf-8"))

    return hasher.hexdigest()[:24]


def get_embedding_cache_path(cache_dir: str | Path, cache_key: str) -> Path:
    """
    Build the cache file path for a given cache key.
    """
    return Path(cache_dir) / f"{cache_key}.pkl"


def load_embedding_cache(cache_dir: str | Path, cache_key: str) -> list[dict] | None:
    """
    Load cached embedded documents if the cache file exists.
    """
    cache_path = get_embedding_cache_path(cache_dir, cache_key)

    if not cache_path.exists():
        return None

    with cache_path.open("rb") as file:
        return pickle.load(file)


def save_embedding_cache(
    cache_dir: str | Path,
    cache_key: str,
    documents: list[dict],
) -> Path:
    """
    Save embedded documents to a local pickle cache file.
    """
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    cache_path = get_embedding_cache_path(cache_dir_path, cache_key)

    with cache_path.open("wb") as file:
        pickle.dump(documents, file)

    return cache_path