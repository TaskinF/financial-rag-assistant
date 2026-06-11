# Week 02 - Retrieval Layer

## Day 8 - Retriever Abstraction and Real BGE-M3 Retrieval Test

### What I implemented
- Added a retriever abstraction for the RAG layer.
- Implemented `VectorStoreRetriever` as a thin wrapper around vector store similarity search.
- Added validation for empty queries and invalid `top_k` values.
- Created a manual retrieval script using real BGE-M3 embeddings.
- Connected ingestion, BGE-M3 embeddings, in-memory vector store and retriever into the first end-to-end retrieval flow.
- Tested the pipeline on a real fund report PDF and retrieved top-k chunks for a financial query.

### Key technical decisions
- Separated the retriever layer from the vector store implementation.
- Kept retrieval orchestration independent from embedding generation and LLM answer generation.
- Used BGE-M3 dense embeddings through the existing embedding interface for realistic multilingual financial document retrieval.
- Added a `--max-chunks` option to the manual retrieval script to make smoke testing feasible on large PDFs.
- Added timing logs to measure ingestion, model loading, embedding/indexing and retrieval latency.
- Prioritized real financial document retrieval quality over a simple demo-only flow.

### Why it matters
Retrieval quality is the foundation of a useful RAG system. If the correct chunks are not retrieved, the LLM cannot produce reliable source-grounded answers. Testing retrieval with a real financial PDF revealed practical bottlenecks such as table extraction noise, indexing latency and the need for embedding persistence or vector index caching.
- I separated the retriever layer from the vector store and tested the first real BGE-M3 retrieval flow on a financial fund report. The system successfully retrieved portfolio-related chunks, but the experiment also showed that financial RAG quality depends heavily on PDF parsing, chunking strategy, embedding latency and persistent indexing.