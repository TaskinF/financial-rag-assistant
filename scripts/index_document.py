import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.document_indexing_service import DocumentIndexingService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index PDF documents into the document registry and ChromaDB."
    )
    parser.add_argument(
        "--pdf-path",
        help="Path to the PDF file to index.",
    )
    parser.add_argument(
        "--document-id",
        help="Optional explicit document identifier.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size used during indexing.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap used during indexing.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        help="Optional inclusive start page filter.",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        help="Optional inclusive end page filter.",
    )
    parser.add_argument(
        "--list-documents",
        action="store_true",
        help="List registered documents instead of indexing a PDF.",
    )
    args = parser.parse_args()

    if not args.list_documents and not args.pdf_path:
        parser.error("--pdf-path is required unless --list-documents is used")

    return args


def print_documents(documents: list[dict]) -> None:
    if not documents:
        print("No indexed documents found in the registry.")
        return

    for document in documents:
        print("Document")
        print(f"document_id: {document.get('document_id')}")
        print(f"filename: {document.get('filename')}")
        print(f"chunk_count: {document.get('chunk_count')}")
        print(f"indexed_at: {document.get('indexed_at')}")
        print(f"status: {document.get('status')}")
        print()


def main() -> None:
    args = parse_args()
    service = DocumentIndexingService()

    try:
        if args.list_documents:
            print_documents(service.list_documents())
            return

        result = service.index_pdf(
            pdf_path=args.pdf_path,
            document_id=args.document_id,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            start_page=args.start_page,
            end_page=args.end_page,
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("Indexed document")
    print(f"document_id: {result.get('document_id')}")
    print(f"filename: {result.get('filename')}")
    print(f"chunk_count: {result.get('chunk_count')}")
    print(f"collection_name: {result.get('collection_name')}")
    print(f"indexed_at: {result.get('indexed_at')}")
    print(f"status: {result.get('status')}")


if __name__ == "__main__":
    main()
