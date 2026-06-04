# Week 01 - RAG Basics

## Day 1 - Project Initialization

### What I implemented
- Initialized the `financial-rag-assistant` repository.
- Created a production-oriented Python project skeleton.
- Added FastAPI entrypoint, configuration module, document loader placeholder, chunking module, vector store placeholder and RAG pipeline placeholder.
- Added basic test structure, README, `.env.example`, `.gitignore` and `requirements.txt`.
- Completed the first GitHub commit and push.

### Key technical decisions
- Started with a modular project structure to keep ingestion, chunking, vector storage and RAG orchestration separated.
- Added FastAPI from the beginning to keep the project aligned with production serving.
- Added tests early to build the habit of validating each pipeline component.

### Why it matters
A clean project skeleton makes the RAG system easier to extend, test and explain in interviews.

---

## Day 2 - PDF Loader

### What I implemented
- Implemented the first step of the RAG ingestion pipeline: PDF text extraction.
- Added a `load_pdf_pages(file_path: str) -> list[dict]` function.
- Extracted PDF content page by page using PyMuPDF.
- Preserved metadata such as `source_file` and `page_number`.
- Added basic validation for missing files and non-PDF inputs.

### Key technical decisions
- Used PyMuPDF because it is lightweight, fast and suitable for page-level PDF text extraction.
- Stored page-level metadata to support traceability and source-grounded answers.
- Skipped empty pages to avoid noisy chunks in later stages.
- Added explicit error handling to make the ingestion pipeline more reliable and testable.

### Why it matters
High-quality text extraction is the foundation of a RAG system. If the extracted text is noisy or incomplete, downstream chunking, embedding, retrieval and final LLM answers will also degrade.