from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from requests.exceptions import RequestException

from app.api.schemas import (
    DocumentAskRequest,
    DocumentAskResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.services.document_indexing_service import DocumentIndexingService
from app.services.rag_service import RAGService


MAX_UPLOAD_SIZE = 50 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


@lru_cache
def get_document_indexing_service() -> DocumentIndexingService:
    """Return the shared document indexing service."""
    return DocumentIndexingService()


@lru_cache
def get_rag_service() -> RAGService:
    """Return the shared RAG service."""
    return RAGService()


@router.post("/upload", response_model=DocumentResponse)
def upload_document(
    file: UploadFile = File(...),
    document_id: str | None = Form(None),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    start_page: int | None = Form(None),
    end_page: int | None = Form(None),
    service: DocumentIndexingService = Depends(get_document_indexing_service),
) -> DocumentResponse:
    """Upload and index a PDF document."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="PDF filename is required")

    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if file.content_type and file.content_type.lower() != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Uploaded file content type must be application/pdf",
        )

    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
            temporary_path = Path(temporary_file.name)
            total_bytes = 0

            while content := file.file.read(UPLOAD_CHUNK_SIZE):
                total_bytes += len(content)
                if total_bytes > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded PDF exceeds the 50 MB size limit",
                    )
                temporary_file.write(content)

        record = service.index_pdf(
            pdf_path=str(temporary_path),
            document_id=document_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_page=start_page,
            end_page=end_page,
        )
        return DocumentResponse(**record)
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="PDF file not found") from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Failed to upload and index the document",
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@router.get("", response_model=DocumentListResponse)
def list_documents(
    service: DocumentIndexingService = Depends(get_document_indexing_service),
) -> DocumentListResponse:
    """List all indexed documents."""
    try:
        documents = service.list_documents()
        return DocumentListResponse(documents=documents, total=len(documents))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Failed to list indexed documents",
        ) from error


@router.post("/{document_id}/ask", response_model=DocumentAskResponse)
def ask_document_question(
    document_id: str,
    request: DocumentAskRequest,
    service: RAGService = Depends(get_rag_service),
) -> DocumentAskResponse:
    """Answer a question using one indexed document."""
    try:
        return service.answer_document_question(document_id, request)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Document not found") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RequestException as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is not reachable. Make sure Ollama is running and "
                "the selected model is available."
            ),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Failed to answer the document question",
        ) from error
