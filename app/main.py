from fastapi import FastAPI, HTTPException

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
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
