# Financial Document RAG Assistant

Local multi-document RAG assistant built for financial PDF reports. It combines BGE-M3 retrieval, persistent ChromaDB indexing, local Ollama answer generation, FastAPI, Streamlit, and retrieval evaluation utilities.

## Key Features

- PDF ingestion with page-level metadata
- Conservative financial text cleaning
- BGE-M3 dense retrieval
- Persistent ChromaDB vector storage
- Document-specific retrieval with metadata filtering
- Focused answer context
- System-managed source metadata
- Local Ollama LLM support
- FastAPI upload, document listing, and question-answering endpoints
- Streamlit document QA interface
- CLI demo
- Retrieval evaluation with Precision@K, Recall@K, MRR, NDCG

## Architecture

```text
Streamlit UI
    | HTTP
    v
FastAPI
    |
    v
Document Registry + ChromaDB
    |
    v
BGE-M3 Retrieval
    |
    v
Ollama Answer Generation
```

## Installation

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

```bash
python -m pip install -r requirements.txt
```

## Run API

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Streamlit UI

Start the FastAPI backend:

```bash
uvicorn app.main:app --reload
```

In a separate terminal, start the UI:

```bash
streamlit run ui/app.py
```

Open:

```text
http://localhost:8501
```

For local Ollama answers, make sure Ollama is running and pull the model:

```bash
ollama pull gemma3:4b
```

UI workflow:

1. Upload a PDF.
2. Select **Upload and Index**.
3. Choose an indexed document.
4. Enter a question.
5. Select the Ollama or fake provider.
6. Review the answer and source pages.

## Run Tests

```bash
python -m pytest
```

## CLI Usage

```bash
python -m scripts.ask_pdf --question "Fonun yonetim ucreti nedir?" --llm-provider ollama --llm-model gemma3:4b
```

## Evaluation

PDF files are local runtime data and are not committed. Place the evaluation PDF under `data/raw/` before running the command.

```bash
python -m scripts.evaluate_rag --pdf-path data/raw/KPC_2026.05.pdf --eval-path eval/rag_eval_questions.json --start-page 1 --end-page 3 --top-k 3 --use-cache
```

Example result:

```text
Pass rate: 100.00%
Average Precision@3: 0.7500
Average Recall@3: 1.0000
Average MRR@3: 1.0000
Average NDCG@3: 0.9599
```

## Example Output

```text
Question: Fonun yonetim ucreti nedir?
Answer: Fonun yonetim ucreti 7.125.856,54 TL'dir.
Source: KPC_2026.05.pdf, page 3
```

## Current Limitations

- PDF table extraction can be noisy
- CPU-based BGE-M3 indexing can be slow for large PDFs
- PDF indexing runs synchronously during the upload request
- The JSON document registry targets local, single-process usage
- Evaluation set is small
- Not financial advice

## Next Steps

- Larger evaluation set
- Background indexing jobs with progress reporting
- Better table parsing and OCR support
- Hybrid retrieval and reranking
- Authentication and production observability
- Agentic financial assistant
