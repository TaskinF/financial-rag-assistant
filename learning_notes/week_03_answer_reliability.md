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
