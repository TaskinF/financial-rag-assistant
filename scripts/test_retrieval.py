from app.vectorstore.embeddings import BGEFlagEmbeddingModel
from app.vectorstore.in_memory_store import InMemoryVectorStore


def main() -> None:
    embedding_model = BGEFlagEmbeddingModel()
    store = InMemoryVectorStore(embedding_model=embedding_model)

    chunks = [
        {
            "chunk_id": "report_q1.pdf_p1_c0",
            "text": "Revenue increased by 12.5% in the first quarter of 2024.",
            "source_file": "report_q1.pdf",
            "page_number": 1,
            "chunk_index": 0,
        },
        {
            "chunk_id": "report_q1.pdf_p2_c0",
            "text": "Operating expenses decreased due to lower logistics costs.",
            "source_file": "report_q1.pdf",
            "page_number": 2,
            "chunk_index": 0,
        },
        {
            "chunk_id": "report_q1.pdf_p3_c0",
            "text": "Net profit margin improved compared with the previous year.",
            "source_file": "report_q1.pdf",
            "page_number": 3,
            "chunk_index": 0,
        },
        {
            "chunk_id": "report_q1.pdf_p4_c0",
            "text": "Cash flow from operations remained stable during the period.",
            "source_file": "report_q1.pdf",
            "page_number": 4,
            "chunk_index": 0,
        },
    ]

    store.add_documents(chunks)

    query = "Which section talks about revenue growth?"
    results = store.similarity_search(query, top_k=3)

    print("Query:", query)
    print()

    for index, result in enumerate(results, start=1):
        print(f"Result {index}")
        print("chunk_id:", result["chunk_id"])
        print("score:", result["score"])
        print("source_file:", result["source_file"])
        print("page_number:", result["page_number"])
        print("text:", result["text"])
        print()


if __name__ == "__main__":
    main()