import os
from dataclasses import dataclass


@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "Financial Document RAG Assistant")
    environment: str = os.getenv("ENVIRONMENT", "development")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))


settings = Settings()