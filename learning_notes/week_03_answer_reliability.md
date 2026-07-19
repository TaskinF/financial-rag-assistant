# Week 03 - Answer Reliability

---

## Day 15 - Focused Answer Context

### What I implemented

* Tested Ollama answer generation with focused retrieval context.
* Limited answer generation to the most relevant chunks.
* Kept sources managed by the system instead of the LLM.

### Key technical decisions

* Used top-ranked retrieved chunks for answer generation.
* Separated LLM answer from source metadata.
* Continued using Ollama as the local LLM provider.

### Why it matters

Financial PDFs contain many similar numbers. Passing too much context can confuse the LLM and cause wrong numeric answers. Focused context improves answer reliability.

- I improved financial RAG answer reliability by limiting the LLM context to the most relevant chunks and keeping citations system-managed. This helped the local LLM return the correct management fee value from the fund report.

## Day 16 - Retrieval Evaluation v0

### What I implemented

* Added a small retrieval evaluation set.
* Built a retrieval evaluation script.
* Measured Hit@K, Precision@K, Recall@K, MRR@K and NDCG@K.

### Key technical decisions

* Evaluated retrieval before LLM answer quality.
* Used expected page and expected keywords as binary relevance signals.
* Kept the evaluation lightweight and easy to extend.

### Why it matters

RAG quality depends on retrieving the right context before generation. Retrieval metrics make the system measurable instead of relying only on manual tests.

- I added retrieval evaluation with standard ranking metrics such as Precision@K, Recall@K, MRR and NDCG, making the financial RAG pipeline objectively measurable.


## Day 17 - FastAPI Ask Endpoint

### What I implemented
- Added FastAPI request and response schemas.
- Added a RAG service layer for question answering.
- Added `/health` and `/ask` endpoints.
- Tested the API with Ollama.

### Key technical decisions
- Kept API logic separate from RAG orchestration.
- Used focused answer context with `answer_top_k`.
- Returned answer, sources and retrieval metadata from the API.

### Why it matters
The project moved from script-based testing to an API-based RAG assistant. This makes the system easier to test, demo and extend.
- I exposed the financial RAG pipeline through a FastAPI `/ask` endpoint and returned source-grounded answers with document metadata.


## Day 18 - API Polish and CLI Demo

### What I implemented
- Added answer cleanup for LLM outputs.
- Added a CLI script for asking questions over financial PDFs.
- Updated README usage examples.

### Key technical decisions
- Cleaned LLM formatting artifacts before returning API responses.
- Reused the existing RAG service in the CLI.
- Kept API and CLI outputs consistent.

### Why it matters
The project is now easier to demo from both API and terminal.
- I polished the financial RAG assistant by cleaning model outputs and adding a CLI demo on top of the FastAPI-based RAG service.

## Day 19 - Final Documentation

### What I implemented
- Polished the README for GitHub usage.
- Added a final project checklist.

### Key technical decisions
- Focused on making the project easy to explain and demo.
- Documented architecture, evaluation results and limitations.
- Kept the final documentation concise.

### Why it matters
A strong project needs both working code and a clear explanation.


## Day 20 - Persistent Vector Store with ChromaDB

### What I implemented
- Added ChromaDB as a persistent vector store.
- Created a ChromaVectorStore wrapper.
- Stored chunk text, embeddings and metadata in a persistent collection.
- Added vector store selection to the retrieval script.

### Key technical decisions
- Kept BGE-M3 as the embedding model.
- Passed embeddings directly to Chroma instead of using Chroma's default embedding function.
- Used upsert to avoid duplicate chunks.
- Preserved source metadata for citation support.

### Why it matters
The project moved from in-memory retrieval to persistent vector storage. This makes repeated document querying faster and prepares the system for multi-document support.
- I integrated ChromaDB as a persistent vector database and kept the embedding/retrieval pipeline provider-independent.


## Day 21 - Document Registry and Multi-PDF Indexing

### What I implemented
- Added a JSON document registry.
- Added document_id-based Chroma metadata filtering.
- Added a PDF indexing service and CLI.

### Key technical decisions
- Used one Chroma collection with document_id metadata.
- Stored uploaded PDFs outside Git tracking.
- Used document_id filters to prevent cross-document retrieval.

### Why it matters
The project moved from a single-PDF flow to persistent multi-document indexing.
- I added document registry and multi-PDF indexing support, making the financial RAG assistant closer to a usable document QA product.