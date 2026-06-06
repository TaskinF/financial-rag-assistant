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
- I started the RAG project with a modular, production-oriented structure instead of a notebook-only prototype.

---

PDF / doküman
   ↓
Text extraction
   ↓
Cleaning
   ↓
Chunking
   ↓
Embedding
   ↓
Vector DB
   ↓
Retrieval
   ↓
LLM answer

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
- In the ingestion layer of my RAG pipeline, I extracted PDF content page by page and preserved source metadata such as file name and page number. This enabled traceability and later source-grounded answer generation.

---

## Day 3 - Text Cleaning

### What I implemented
- Added a text cleaning utility for the RAG ingestion pipeline.
- Implemented `clean_text(text: str | None) -> str`.
- Normalized tabs, newlines and repeated whitespace.
- Preserved financial symbols, numbers, percentages, dates and punctuation.
- Added unit tests for whitespace normalization and financial information preservation.

### Key technical decisions
- Kept the cleaning logic conservative to avoid losing important financial information.
- Separated text cleaning from PDF loading to keep the ingestion pipeline modular and testable.
- Used simple whitespace normalization instead of aggressive regex-based text removal.
- Added `None` input handling to make the pipeline more robust.

### Why it matters
Text cleaning directly affects chunking, embedding quality and retrieval accuracy. In financial documents, overly aggressive cleaning can remove critical information such as percentages, monetary values and dates.
- I designed the text cleaning layer conservatively because financial RAG systems must preserve numerical and symbolic information. The goal was to reduce noise without damaging the semantic and quantitative content needed for reliable retrieval.