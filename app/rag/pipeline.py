from app.processing.chunking import chunk_text


class RAGPipeline:
    def __init__(self) -> None:
        pass

    def process_document(self, text: str) -> list[str]:
        return chunk_text(text)

    def answer(self, question: str) -> str:
        # Placeholder only: retrieval + generation will be added later.
        return "Not implemented yet."