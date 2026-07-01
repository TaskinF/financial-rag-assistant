import argparse
import json
from pathlib import Path

from app.processing.ingestion import build_chunks_from_pdf
from app.rag.retrieval_evaluation import evaluate_retrieval_results
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
        description="Run retrieval evaluation on a PDF and an evaluation question set."
    )
    parser.add_argument("--pdf-path", required=True, help="Path to the PDF file.")
    parser.add_argument(
        "--eval-path",
        default="eval/rag_eval_questions.json",
        help="Path to the retrieval evaluation JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved results to evaluate.",
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
        "--max-chunks",
        type=int,
        default=None,
        help="If provided, only the first N chunks will be indexed.",
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
        "--use-cache",
        action="store_true",
        help="Use embedding cache if available.",
    )
    parser.add_argument(
        "--cache-dir",
        default="artifacts/embedding_cache",
        help="Directory for embedding cache files.",
    )
    return parser.parse_args()


def filter_chunks_by_page(
    chunks: list[dict],
    start_page: int | None = None,
    end_page: int | None = None,
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

    return filtered_chunks


def load_eval_items(eval_path: Path) -> list[dict]:
    with eval_path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()

    pdf_path = Path(args.pdf_path)
    eval_path = Path(args.eval_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    chunks = build_chunks_from_pdf(
        str(pdf_path),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = filter_chunks_by_page(
        chunks,
        start_page=args.start_page,
        end_page=args.end_page,
    )

    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]

    embedding_model = BGEFlagEmbeddingModel(use_fp16=False)
    vector_store = InMemoryVectorStore(embedding_model=embedding_model)

    if args.use_cache:
        cache_key = build_embedding_cache_key(
            chunks=chunks,
            model_name=embedding_model.model_name,
        )
        cached_documents = load_embedding_cache(args.cache_dir, cache_key)

        if cached_documents is not None:
            print(f"Embedding cache hit: {cache_key}")
            vector_store.add_embedded_documents(cached_documents)
        else:
            print(f"Embedding cache miss: {cache_key}")
            vector_store.add_documents(chunks)
            cache_path = save_embedding_cache(
                cache_dir=args.cache_dir,
                cache_key=cache_key,
                documents=vector_store.get_documents(),
            )
            print(f"Saved embedding cache to: {cache_path}")
    else:
        vector_store.add_documents(chunks)

    retriever = VectorStoreRetriever(vector_store=vector_store)
    eval_items = load_eval_items(eval_path)

    total = len(eval_items)
    passed = 0
    failed = 0
    precision_values: list[float] = []
    recall_values: list[float] = []
    mrr_values: list[float] = []
    ndcg_values: list[float] = []

    for item in eval_items:
        question = item["question"]
        expected_page = item["expected_page"]
        expected_keywords = item["expected_keywords"]

        results = retriever.retrieve(question, top_k=args.top_k)
        evaluation = evaluate_retrieval_results(
            results=results,
            expected_page=expected_page,
            expected_keywords=expected_keywords,
            k=args.top_k,
        )
        retrieved_pages = [result.get("page_number") for result in results]

        print(f"ID: {item['id']}")
        print(f"Question: {question}")
        print(f"Expected page: {expected_page}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"Relevance by rank: {evaluation['relevance_by_rank']}")
        print(f"Hit@{args.top_k}: {evaluation['hit_at_k']:.4f}")
        print(f"Precision@{args.top_k}: {evaluation['precision_at_k']:.4f}")
        print(f"Recall@{args.top_k}: {evaluation['recall_at_k']:.4f}")
        print(f"MRR@{args.top_k}: {evaluation['mrr_at_k']:.4f}")
        print(f"NDCG@{args.top_k}: {evaluation['ndcg_at_k']:.4f}")
        print(f"Status: {evaluation['status']}")
        print()

        if evaluation["status"] == "PASS":
            passed += 1
        else:
            failed += 1

        precision_values.append(evaluation["precision_at_k"])
        recall_values.append(evaluation["recall_at_k"])
        mrr_values.append(evaluation["mrr_at_k"])
        ndcg_values.append(evaluation["ndcg_at_k"])

    pass_rate = (passed / total * 100.0) if total else 0.0
    avg_precision = (sum(precision_values) / total) if total else 0.0
    avg_recall = (sum(recall_values) / total) if total else 0.0
    avg_mrr = (sum(mrr_values) / total) if total else 0.0
    avg_ndcg = (sum(ndcg_values) / total) if total else 0.0

    print("Summary")
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass rate: {pass_rate:.2f}%")
    print()
    print(f"Average Precision@{args.top_k}: {avg_precision:.4f}")
    print(f"Average Recall@{args.top_k}: {avg_recall:.4f}")
    print(f"Average MRR@{args.top_k}: {avg_mrr:.4f}")
    print(f"Average NDCG@{args.top_k}: {avg_ndcg:.4f}")


if __name__ == "__main__":
    main()
