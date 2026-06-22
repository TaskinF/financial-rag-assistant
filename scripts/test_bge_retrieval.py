import argparse
import time
from pathlib import Path

import requests

from app.llm.llm_client import FakeLLMClient
from app.llm.ollama_client import OllamaLLMClient
from app.processing.ingestion import build_chunks_from_pdf
from app.rag.answer_generator import generate_answer
from app.rag.context_builder import build_context, extract_sources
from app.rag.retriever import VectorStoreRetriever
from app.vectorstore.embedding_cache import (
    build_embedding_cache_key,
    load_embedding_cache,
    save_embedding_cache,
)
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
        "--start-page",
        type=int,
        default=None,
        help="Only keep chunks with page_number >= start_page.",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Only keep chunks with page_number <= end_page.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Only keep chunks from the first N pages.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="If provided, only the first N chunks will be indexed.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="Number of characters to show in each result preview.",
    )
    parser.add_argument(
        "--context-max-chars",
        type=int,
        default=4000,
        help="Maximum number of characters for the LLM-ready context preview.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="artifacts/embedding_cache",
        help="Directory for storing embedding cache files.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache usage.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print LLM-ready context preview after retrieval.",
    )
    parser.add_argument(
        "--show-answer",
        action="store_true",
        help="Run answer generation preview on retrieved results.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["fake", "ollama"],
        default="fake",
        help="LLM provider to use for answer preview.",
    )
    parser.add_argument(
        "--llm-model",
        default="gemma3:4b",
        help="LLM model name for answer preview.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Base URL for the Ollama server.",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Use fp16 for BGE-M3 model loading. Default is False for CPU safety.",
    )
    return parser.parse_args()


def filter_chunks_by_page(
    chunks: list[dict],
    start_page: int | None = None,
    end_page: int | None = None,
    max_pages: int | None = None,
) -> list[dict]:
    filtered_chunks = chunks

    if start_page is not None:
        filtered_chunks = [
            chunk
            for chunk in filtered_chunks
            if chunk.get("page_number") is not None
            and chunk["page_number"] >= start_page
        ]

    if end_page is not None:
        filtered_chunks = [
            chunk
            for chunk in filtered_chunks
            if chunk.get("page_number") is not None
            and chunk["page_number"] <= end_page
        ]

    if max_pages is not None:
        filtered_chunks = [
            chunk
            for chunk in filtered_chunks
            if chunk.get("page_number") is not None
            and chunk["page_number"] <= max_pages
        ]

    return filtered_chunks


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

    total_chunks_before_filtering = len(chunks)

    print("PDF:", pdf_path.name)
    print("Total chunks before page filtering:", total_chunks_before_filtering)
    print(f"Ingestion/chunk generation took {ingestion_elapsed:.2f} seconds.")
    print()

    chunks = filter_chunks_by_page(
        chunks,
        start_page=args.start_page,
        end_page=args.end_page,
        max_pages=args.max_pages,
    )

    print("Chunks after page filtering:", len(chunks))

    if args.start_page is not None or args.end_page is not None:
        start_label = args.start_page if args.start_page is not None else "beginning"
        end_label = args.end_page if args.end_page is not None else "end"
        print(f"Using pages: {start_label}-{end_label}")
    elif args.max_pages is not None:
        print(f"Using pages: 1-{args.max_pages}")

    print()

    if not chunks:
        print("No chunks left after page filtering.")
        return

    if args.max_chunks is not None:
        chunks = chunks[:args.max_chunks]
        print(f"Using first {args.max_chunks} chunks for smoke test.")
        print()

    cache_key = build_embedding_cache_key(
        chunks=chunks,
        model_name="BAAI/bge-m3",
    )

    model_start = time.perf_counter()
    embedding_model = BGEFlagEmbeddingModel(use_fp16=args.use_fp16)
    model_elapsed = time.perf_counter() - model_start

    print(f"BGE-M3 model loading took {model_elapsed:.2f} seconds.")
    print()
    print("Indexing chunks with BGE-M3 embeddings. This may take a while on CPU...")

    index_start = time.perf_counter()
    vector_store = InMemoryVectorStore(embedding_model=embedding_model)

    cached_documents = None
    if not args.no_cache:
        cached_documents = load_embedding_cache(args.cache_dir, cache_key)

    if cached_documents is not None:
        print(f"Embedding cache hit: {cache_key}")
        vector_store.add_embedded_documents(cached_documents)
    else:
        if args.no_cache:
            print("Embedding cache disabled.")
        else:
            print(f"Embedding cache miss: {cache_key}")

        vector_store.add_documents(chunks)

        if not args.no_cache:
            embedded_documents = vector_store.get_documents()
            cache_path = save_embedding_cache(
                cache_dir=args.cache_dir,
                cache_key=cache_key,
                documents=embedded_documents,
            )
            print(f"Saved embedding cache to: {cache_path}")

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

    if not results:
        print("No retrieval results found.")
        return

    for rank, result in enumerate(results, start=1):
        preview_text = " ".join(result["text"].split())[:args.preview_chars]

        print("--------------------------------------------------")
        print(f"Rank: {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source File: {result['source_file']}")
        print(f"Page Number: {result['page_number']}")
        print(f"Text Preview: {preview_text}")
        print()

    if args.show_context:
        context = build_context(results, max_chars=args.context_max_chars)
        sources = extract_sources(results)

        print("==================================================")
        print("LLM-ready context preview")
        print("==================================================")
        print(context)
        print()

        print("Sources")
        for index, source in enumerate(sources, start=1):
            print(f"{index}. {source}")
        print()

    if args.show_answer:
        if args.llm_provider == "fake":
            llm_client = FakeLLMClient(response="Fake source-grounded answer preview")
        else:
            llm_client = OllamaLLMClient(
                base_url=args.ollama_base_url,
                model=args.llm_model,
            )

        try:
            answer_payload = generate_answer(
                question=args.query,
                retrieved_chunks=results,
                llm_client=llm_client,
                max_context_chars=args.context_max_chars,
            )
        except requests.RequestException:
            print("Ollama is not reachable. Make sure Ollama is running and the selected model is pulled.")
            return

        print("==================================================")
        print("Answer generation preview")
        print("==================================================")
        print("Provider:", args.llm_provider)
        print("Model:", args.llm_model if args.llm_provider == "ollama" else "fake")
        print("Answer:", answer_payload["answer"])
        print("Sources:")
        for index, source in enumerate(answer_payload["sources"], start=1):
            print(f"{index}. {source}")
        print("Context length:", len(answer_payload["context"]))
        print()


if __name__ == "__main__":
    main()