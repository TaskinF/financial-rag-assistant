# Financial Document RAG Assistant

A production-oriented starter skeleton for a financial document RAG application in Python.

## Goals
- Load and process financial PDFs
- Split text into chunks
- Store embeddings in a vector database
- Build a retrieval pipeline for question answering

## Current Scope
This version only includes:
- FastAPI app entrypoint
- Config with environment variables
- PDF loader placeholder
- Text chunking module
- Vector store placeholder
- RAG pipeline placeholder
- Basic pytest test

## Run
```bash
uvicorn app.main:app --reload