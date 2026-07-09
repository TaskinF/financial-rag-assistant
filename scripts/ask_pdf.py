import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.schemas import AskRequest
from app.services.rag_service import RAGService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question to the Financial RAG assistant from the terminal."
    )
    parser.add_argument(
        "--pdf-path",
        default="data/raw/KPC_2026.05.pdf",
        help="Path to the PDF file.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask about the PDF content.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Inclusive start page for retrieval.",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=3,
        help="Inclusive end page for retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks.",
    )
    parser.add_argument(
        "--answer-top-k",
        type=int,
        default=2,
        help="Number of top chunks to use for answer generation.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["fake", "ollama"],
        default="fake",
        help="LLM provider to use.",
    )
    parser.add_argument(
        "--llm-model",
        default="gemma3:4b",
        help="LLM model name.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=4000,
        help="Maximum context length for answer generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        request = AskRequest(
            question=args.question,
            pdf_path=args.pdf_path,
            start_page=args.start_page,
            end_page=args.end_page,
            top_k=args.top_k,
            answer_top_k=args.answer_top_k,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            max_context_chars=args.max_context_chars,
        )

        response = RAGService().answer_question(request)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("Question:")
    print(response.question)
    print()

    print("Answer:")
    print(response.answer)
    print()

    print("Sources:")
    if not response.sources:
        print("- none")
        return

    for source in response.sources:
        source_file = source.source_file
        page_number = source.page_number
        chunk_id = source.chunk_id
        score = source.score

        if score is not None:
            print(
                f"- {source_file}, page {page_number}, "
                f"chunk_id={chunk_id}, score={score:.4f}"
            )
        else:
            print(f"- {source_file}, page {page_number}, chunk_id={chunk_id}")


if __name__ == "__main__":
    main()
