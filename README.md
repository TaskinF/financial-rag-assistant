# Financial Document RAG Assistant

A modular Retrieval-Augmented Generation (RAG) project for financial PDF documents.

The project processes financial reports, extracts and chunks text, generates embeddings with BGE-M3, retrieves relevant document sections, and prepares source-grounded answers with citation-ready metadata.

The goal is to build a practical financial document assistant, not just a simple RAG demo.

---

## Features

* PDF text extraction with page-level metadata
* Conservative text cleaning for financial documents
* Metadata-preserving text chunking
* BGE-M3 embedding support via FlagEmbedding
* In-memory vector store with cosine similarity search
* Retriever abstraction
* Page-aware retrieval for large financial PDFs
* Embedding cache for faster repeated retrieval tests
* LLM-ready context builder
* Source extraction for citation support
* Prompt and answer generation layer with fake LLM client for testing
* Unit tests with pytest

---

## Project Structure

```text
financial-rag-assistant/
├── app/
│   ├── loaders/          # PDF loading
│   ├── processing/       # Cleaning, chunking, ingestion
│   ├── vectorstore/      # Embeddings, vector store, cache
│   ├── rag/              # Retriever, context, prompt, answer generation
│   ├── llm/              # LLM client abstraction
│   ├── config.py
│   └── main.py
│
├── data/raw/             # Local PDF files
├── artifacts/            # Local generated artifacts
├── scripts/              # Manual test scripts
├── tests/                # Unit tests
├── learning-notes/       # Development notes
├── README.md
├── requirements.txt
└── .env.example
```

---

## RAG Pipeline

```text
PDF
→ Text Extraction
→ Cleaning
→ Chunking
→ Embeddings
→ Vector Search
→ Retrieval
→ Context Building
→ Prompt Building
→ Answer Generation
```

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the FastAPI App

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

## Run Tests

```bash
pytest
```

---

## Manual Retrieval Test

Place a financial PDF under:

```text
data/raw/
```

Example:

```text
data/raw/KPC_2026.05.pdf
```

Run a basic retrieval test:

```bash
python -m scripts.test_bge_retrieval --pdf-path data/raw/KPC_2026.05.pdf --query "Fonun yönetim ücreti nedir?"
```

Run with page filtering:

```bash
python -m scripts.test_bge_retrieval --pdf-path data/raw/KPC_2026.05.pdf --start-page 1 --end-page 3 --query "Fonun yönetim ücreti nedir?"
```

Show LLM-ready context:

```bash
python -m scripts.test_bge_retrieval --pdf-path data/raw/KPC_2026.05.pdf --start-page 1 --end-page 3 --query "Fonun yönetim ücreti nedir?" --show-context
```

Show fake answer generation preview:

```bash
python -m scripts.test_bge_retrieval --pdf-path data/raw/KPC_2026.05.pdf --start-page 1 --end-page 3 --query "Fonun yönetim ücreti nedir?" --show-answer
```

---

## Example Observation

In a real 177-page fund report, the system retrieved the relevant chunk for:

```text
Fonun yönetim ücreti nedir?
```

The retrieved chunk contained:

```text
Fon Yönetim Ücreti 7.125.856,54 0,1836 %
```

This showed that BGE-M3 can retrieve precise financial information, while also revealing that table-heavy financial PDFs require careful parsing, chunking and metadata handling.

---

## Current Limitations

* PDF table extraction can be noisy
* In-memory vector store is for local development only
* Real LLM provider integration is not finalized yet
* Automated retrieval evaluation is not implemented yet
* UI is not implemented yet

---

## Roadmap

* Add real LLM provider support
* Add citation-based answer generation
* Add retrieval evaluation dataset
* Add persistent FAISS or Chroma vector store
* Improve table-heavy PDF parsing
* Add FastAPI `/ingest` and `/ask` endpoints
* Add simple UI
* Add Docker support

---

## Learning Notes

Development notes and implementation decisions are tracked under:

```text
learning-notes/
```
