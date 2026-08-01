from fastapi import FastAPI, HTTPException
from requests.exceptions import RequestException

from app.api.document_routes import router as document_router
from app.api.schemas import AskRequest, AskResponse
from app.services.rag_service import RAGService

app = FastAPI(title="Financial Document RAG Assistant")
app.include_router(document_router)
rag_service = RAGService()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    try:
        return rag_service.answer_question(request)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="PDF file not found") from error
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
            detail="Failed to answer the question",
        ) from error
