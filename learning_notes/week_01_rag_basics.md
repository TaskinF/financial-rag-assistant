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


## Day 4 - Text Chunking

### What I implemented
- Implemented a basic character-based text chunking utility.
- Added `chunk_text(text, chunk_size, chunk_overlap)` for overlapping text chunks.
- Added `chunk_pages(pages, chunk_size, chunk_overlap)` to preserve page-level metadata.
- Generated deterministic chunk IDs using source file, page number and chunk index.
- Added unit tests for parameter validation, overlap behavior and metadata preservation.

### Key technical decisions
- Started with character-based chunking because it is simple, transparent and easy to test.
- Added chunk overlap to reduce context loss across chunk boundaries.
- Preserved metadata such as `source_file`, `page_number` and `chunk_index` for future source-grounded answers.
- Validated chunking parameters to avoid invalid configurations and infinite loop risks.

### Why it matters
Chunking quality directly affects retrieval quality in RAG systems. If chunks are too large, retrieval becomes noisy. If chunks are too small, the LLM may receive incomplete context. Overlap helps preserve meaning across chunk boundaries.
- I implemented a transparent chunking layer before using framework abstractions. This helped me understand how chunk size, overlap and metadata preservation affect retrieval quality and source-grounded answer generation.


---

## Day 5 - Mini Ingestion Pipeline

### What I implemented
- Added a mini ingestion pipeline for the Financial Document RAG Assistant.
- Implemented `build_chunks_from_pdf(file_path, chunk_size, chunk_overlap)`.
- Orchestrated PDF loading, conservative text cleaning and metadata-preserving chunking.
- Preserved source metadata such as `source_file` and `page_number` throughout the pipeline.
- Added integration-style tests using mocked PDF loading behavior.

### Key technical decisions
- Kept PDF loading, text cleaning and chunking as separate modules to preserve separation of concerns.
- Added an ingestion orchestration layer instead of mixing preprocessing logic inside the FastAPI or RAG layer.
- Used metadata-preserving transformations to support future source-grounded answers.
- Used mocked PDF loading in tests to validate pipeline behavior without depending on real PDF files.

### Why it matters
The ingestion pipeline converts raw documents into retrieval-ready chunks. This is the foundation for embedding generation, vector database indexing and reliable RAG answers. If metadata is lost at this stage, source citation and traceability become difficult later.
- I built a modular ingestion pipeline that transforms PDFs into clean, metadata-preserving chunks. I separated loading, cleaning and chunking responsibilities and added integration-style tests to verify that the pipeline preserves traceability from document to chunk.


---

## Day 6 - BGE-M3 Embedding Interface with FlagEmbedding

### What I implemented
- Added a provider-independent embedding model interface.
- Implemented `EmbeddingModel` with `embed_text` and `embed_documents` methods.
- Added `BGEFlagEmbeddingModel` using `FlagEmbedding` and `BAAI/bge-m3`.
- Configured BGE-M3 to return dense embeddings for the initial vector search pipeline.
- Kept a deterministic `FakeEmbeddingModel` for fast and reliable unit tests.
- Added a manual script to test real BGE-M3 embeddings locally.

### Key technical decisions
- Used `FlagEmbedding` instead of a generic sentence-transformers wrapper because BGE-M3 has native support for dense retrieval, sparse retrieval and ColBERT-style multi-vector interaction.
- Started with dense embeddings only to keep the first vector search implementation simple.
- Preserved a provider-independent interface so the project can later support Ollama, OpenAI or Azure OpenAI without changing the rest of the RAG pipeline.
- Avoided loading the real BGE-M3 model in unit tests to keep tests fast and CI-friendly.

### Why it matters
Embedding is the bridge between text chunks and semantic search. Using BGE-M3 through its native FlagEmbedding interface gives the project a stronger foundation for multilingual financial document retrieval, while keeping the architecture extensible for future reranking and hybrid retrieval.
- I implemented the embedding layer with BGE-M3 through FlagEmbedding instead of relying on a generic embedding wrapper. This gives access to dense, sparse and ColBERT-style representations in the future, while the initial MVP uses dense vectors for simple semantic retrieval.