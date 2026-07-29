# Financial Document RAG Assistant

Local RAG assistant built for financial PDF documents. The project includes BGE-M3 retrieval, local Ollama-based answer generation, a FastAPI API, a CLI demo, and retrieval evaluation utilities.

## Key Features

- PDF ingestion with page-level metadata
- Conservative financial text cleaning
- BGE-M3 dense retrieval
- Embedding cache
- Focused answer context
- Local Ollama LLM support
- FastAPI `/ask` endpoint
- CLI demo
- Retrieval evaluation with Precision@K, Recall@K, MRR, NDCG

## Architecture

```text
Streamlit UI
    ↓ HTTP
FastAPI
    ↓
Document Registry + ChromaDB
    ↓
BGE-M3 Retrieval
    ↓
Ollama Answer Generation
```

## Installation

```bash
python -m venv venv
```

```bash
pip install -r requirements.txt
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

## CLI Usage

```bash
python -m scripts.ask_pdf --question "Fonun yonetim ucreti nedir?" --llm-provider ollama --llm-model gemma3:4b
```

## Evaluation

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
- In-memory vector store is for local development
- Evaluation set is small
- Not financial advice

## Next Steps

- Larger evaluation set
- Persistent vector store
- Better table parsing
- Agentic financial assistant
