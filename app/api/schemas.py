from pydantic import BaseModel, field_validator, model_validator


class AskRequest(BaseModel):
    question: str
    pdf_path: str = "data/raw/KPC_2026.05.pdf"
    start_page: int | None = 1
    end_page: int | None = 3
    top_k: int = 3
    answer_top_k: int = 2
    chunk_size: int = 1000
    chunk_overlap: int = 200
    llm_provider: str = "fake"
    llm_model: str = "gemma3:4b"
    max_context_chars: int = 4000

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("question cannot be empty")
        return value.strip()

    @field_validator("top_k", "answer_top_k", "chunk_size", "max_context_chars")
    @classmethod
    def validate_positive_integers(cls, value: int, info) -> int:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be greater than 0")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        return value

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        if value not in {"fake", "ollama"}:
            raise ValueError("llm_provider must be either 'fake' or 'ollama'")
        return value

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "AskRequest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class Source(BaseModel):
    source_file: str | None
    page_number: int | None
    chunk_id: str | None
    score: float | None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    retrieved_count: int
    answer_context_count: int
    llm_provider: str
    llm_model: str
