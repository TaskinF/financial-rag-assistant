class VectorStore:
    def __init__(self) -> None:
        self.documents = []

    def add(self, chunks: list[str]) -> None:
        self.documents.extend(chunks)

    def search(self, query: str) -> list[str]:
        # Placeholder only: similarity search will be added later.
        return []