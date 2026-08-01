# Week 04 - Productization 

---

## Day 22 - Document Upload and QA API

### What I implemented
- Added PDF upload and document listing endpoints.
- Added document_id-based question answering.
- Added upload validation and persistent indexing.

### Key technical decisions
- Used Chroma metadata filters for document isolation.
- Reused persistent indexes instead of reprocessing PDFs per question.
- Kept answer sources system-managed.

### Why it matters
The assistant now supports an end-to-end multi-document API workflow.
- I turned the RAG pipeline into a document product API with upload, persistent indexing, document selection and source-grounded question answering.

---

## Day 23 - Streamlit User Interface

### What I implemented

- Added a Streamlit interface for PDF upload and document selection.
- Added document-specific question answering.
- Displayed answers, sources and retrieval metadata.

### Key technical decisions

- Kept Streamlit as a thin client over FastAPI.
- Reused the existing upload and document QA endpoints.
- Kept indexing and LLM execution in the backend.

### Why it matters

The RAG system became a usable end-to-end document QA product.
- I added a Streamlit frontend over the FastAPI-based multi-document RAG backend, enabling users to upload reports, select documents and view source-grounded answers.

---

## Day 24 - Final Documentation

### What I implemented

- Finalized the README and product workflow.
- Added an end-to-end usage guide.
- Added a final validation checklist.

### Key technical decisions

- Documented only implemented features and verified metrics.
- Kept the usage flow independent from development details.
- Included API, UI and CLI fallback paths.

### Why it matters

The project is now easier to install, use and maintain.
- I finalized the multi-document RAG application with clear architecture, usage documentation and reproducible evaluation steps.
