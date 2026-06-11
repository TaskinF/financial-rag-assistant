import argparse
import time
from pathlib import Path

from app.processing.ingestion import build_chunks_from_pdf
from app.rag.retriever import VectorStoreRetriever
from app.vectorstore.embeddings import BGEFlagEmbeddingModel
from app.vectorstore.in_memory_store import InMemoryVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual BGE-M3 retrieval smoke test for PDF documents."
    )
    parser.add_argument(
        "--pdf-path",
        default="data/raw/KPC_2026.05.pdf",
        help="Path to the PDF file to index.",
    )
    parser.add_argument(
        "--query",
        default="Fonun portföy dağılımı nasıldır?",
        help="Query to run against the indexed chunks.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieval results to return.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size to use during ingestion.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap to use during ingestion.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="If provided, only the first N chunks will be indexed.",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Use fp16 for BGE-M3 model loading. Default is False for CPU safety.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        print("Please provide a valid PDF path and run the script again.")
        return

    ingestion_start = time.perf_counter()
    chunks = build_chunks_from_pdf(
        file_path=str(pdf_path),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    ingestion_elapsed = time.perf_counter() - ingestion_start

    print("PDF:", pdf_path.name)
    print("Total chunks:", len(chunks))
    print(f"Ingestion/chunk generation took {ingestion_elapsed:.2f} seconds.")
    print()

    if args.max_chunks is not None:
        chunks = chunks[:args.max_chunks]
        print(f"Using first {args.max_chunks} chunks for smoke test.")
        print()

    model_start = time.perf_counter()
    embedding_model = BGEFlagEmbeddingModel(use_fp16=args.use_fp16)
    model_elapsed = time.perf_counter() - model_start

    print(f"BGE-M3 model loading took {model_elapsed:.2f} seconds.")
    print()
    print("Indexing chunks with BGE-M3 embeddings. This may take a while on CPU...")

    index_start = time.perf_counter()
    vector_store = InMemoryVectorStore(embedding_model=embedding_model)
    vector_store.add_documents(chunks)
    index_elapsed = time.perf_counter() - index_start

    print(f"Vector store indexing/embedding generation took {index_elapsed:.2f} seconds.")
    print()

    retriever = VectorStoreRetriever(vector_store=vector_store)

    retrieval_start = time.perf_counter()
    results = retriever.retrieve(args.query, top_k=args.top_k)
    retrieval_elapsed = time.perf_counter() - retrieval_start

    print("Query:", args.query)
    print("Retrieved results:", len(results))
    print(f"Retrieval took {retrieval_elapsed:.2f} seconds.")
    print()

    for rank, result in enumerate(results, start=1):
        preview_text = result["text"][:500]

        print(f"Rank {rank}")
        print("score:", result["score"])
        print("chunk_id:", result["chunk_id"])
        print("source_file:", result["source_file"])
        print("page_number:", result["page_number"])
        print("text:", preview_text)
        print()


if __name__ == "__main__":
    main()