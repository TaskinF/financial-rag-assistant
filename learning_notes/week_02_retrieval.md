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

---

## Day 9 - Metadata Hygiene and Page-Aware Retrieval Inspection

### What I implemented

* Cleaned PDF source metadata by storing only the file name instead of the full local file path.
* Improved the manual BGE-M3 retrieval script output formatting.
* Added readable result separators, formatted scores and configurable text preview length.
* Added page-aware filtering support to the retrieval test script.
* Added `--start-page`, `--end-page` and `--max-pages` arguments for testing specific parts of large financial PDFs.
* Kept `--max-chunks` support for fast smoke testing.
* Tested retrieval on a real 177-page KPC fund report PDF.

### Key technical decisions

* Normalized `source_file` metadata to make future source citations cleaner and more user-friendly.
* Kept page filtering inside the manual retrieval script because this is currently an inspection and debugging tool.
* Applied page filtering before chunk limiting so that retrieval tests can focus on specific document sections.
* Used page-aware retrieval inspection because the KPC fund report contains high-value summary information in the first pages and transaction-heavy details in later pages.
* Continued using BGE-M3 dense embeddings for realistic multilingual financial retrieval tests.

### Practical observation

The KPC fund report is a 177-page, table-heavy PDF. The first pages contain high-value summary information such as fund identity, fund portfolio value, fund total value, management fee and portfolio composition. Later pages mostly contain detailed transaction records.

The retrieval test showed that BGE-M3 can correctly retrieve specific financial information. For example, the query "Fonun yönetim ücreti nedir?" returned the chunk containing "Fon Yönetim Ücreti 7.125.856,54 0,1836 %" as the top result.

For broader questions like "Fonun portföy dağılımı nasıldır?", the retrieved chunks were related to portfolio composition, but the text was noisy because the original PDF is table-heavy and difficult to parse.

This showed that the main challenge is not only embedding quality. PDF parsing quality, table extraction, chunking strategy, metadata hygiene and retrieval scope are also critical for financial RAG systems.

### Why it matters

Page-aware retrieval inspection helps avoid embedding and searching unnecessary parts of large documents during debugging. For financial reports, not all pages have equal information value. Summary-level questions should often focus on summary pages, while transaction-level questions may require searching detailed transaction pages.

This also showed that a single full-document index may not always be the best design for financial RAG. A better long-term architecture may include separate retrieval modes such as summary retrieval, transaction-level retrieval and full-document retrieval.
- I improved metadata hygiene and added page-aware retrieval inspection for a real financial RAG pipeline. On a 177-page KPC fund report, I observed that BGE-M3 could retrieve precise information such as management fee, but table-heavy PDF extraction created noisy chunks. This led to the design insight that financial RAG systems should support section-aware or page-aware retrieval, not only full-document semantic search.


---

## Day 10 - Embedding Cache for Retrieval

### What I implemented
- Added local embedding cache support for the manual BGE-M3 retrieval flow.
- Created deterministic cache keys based on chunk content, metadata and embedding model name.
- Added utility functions to save and load embedded documents from local pickle files.
- Extended the in-memory vector store to support pre-computed embeddings.
- Updated the retrieval test script to reuse cached embeddings when available.

### Key technical decisions
- Used chunk content and metadata to generate the cache key instead of relying only on file name.
- Kept the cache local under `artifacts/embedding_cache`.
- Ignored cache files in Git while keeping the directory structure with `.gitkeep`.
- Cached document embeddings, not query embeddings.
- Kept BGE-M3 model loading in the flow because query embedding still requires the model.
- Added `--no-cache` to allow debugging without cache.

### Why it matters
Embedding generation is one of the slowest parts of the RAG pipeline, especially with a strong model like BGE-M3 running on CPU. Caching embeddings prevents repeated computation for the same document chunks and makes retrieval iteration much faster.
- I added a local embedding cache to avoid recomputing BGE-M3 embeddings for the same financial document chunks. This made the retrieval workflow more practical and introduced a production-relevant concept: separating one-time indexing cost from repeated query-time retrieval.


---

## Day 11 - RAG Context Builder

### What I implemented
- Added a context builder for formatting retrieved chunks into an LLM-ready context.
- Added source extraction from retrieved chunks.
- Normalized whitespace in retrieved texts before sending them to the future LLM layer.
- Added configurable context length limiting.
- Added context preview support to the manual BGE-M3 retrieval script.

### Key technical decisions
- Kept context building separate from retrieval and LLM generation.
- Included source file, page number, score and chunk text in the context.
- Added max character limiting to control prompt size.
- Preserved source metadata for future citations.
- Did not add an LLM call yet because the goal was to prepare clean, grounded context first.

### Why it matters
RAG quality depends not only on retrieval but also on how retrieved chunks are presented to the LLM. A clean context format improves answer grounding, makes citations easier and reduces the chance of the model using irrelevant or unstructured input.
- I added a dedicated context builder that converts retrieved financial document chunks into a structured LLM-ready format with source metadata. This prepared the system for source-grounded answer generation while keeping retrieval, context formatting and LLM generation as separate components.