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