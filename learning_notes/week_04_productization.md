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