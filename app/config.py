import os
from dataclasses import dataclass


@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "Financial Document RAG Assistant")
    environment: str = os.getenv("ENVIRONMENT", "development")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")


settings = Settings()