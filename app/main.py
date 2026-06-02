from fastapi import FastAPI

app = FastAPI(title="Financial Document RAG Assistant")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}