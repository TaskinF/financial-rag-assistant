# Product Usage

## Prerequisites

- A Python environment must be ready with project dependencies installed.
- The FastAPI backend must be running.
- The Streamlit UI must be running.
- Ollama must be running.
- The `gemma3:4b` model must be available locally.

Pull the Ollama model:

```bash
ollama pull gemma3:4b
```

Start the backend:

```bash
uvicorn app.main:app --reload
```

Start the UI in a separate terminal:

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` in a browser.

## Upload a Document

1. Open the Streamlit interface.
2. Select a financial PDF.
3. Optionally enter a `document_id`.
4. Select **Upload and Index**.
5. After indexing completes, verify the returned `document_id` and `chunk_count`.

The UI indexes pages 1-3 by default. The page range can be changed under the advanced upload settings.

## Select a Document

1. Select a PDF from the indexed document list.
2. Verify its `filename`, `document_id`, `chunk_count`, and `status`.

## Ask a Question

Example question:

```text
Fonun yönetim ücreti nedir?
```

Provider:

```text
ollama
```

Model:

```text
gemma3:4b
```

Submit the question and verify that the answer includes system-managed source metadata:

- `source_file`
- `page_number`
- `chunk_id`
- retrieval `score`

## Multi-Document Validation

1. Upload at least two different fund reports.
2. Select the first document and ask for its fund name or management fee.
3. Select the second document and ask the same question.
4. Verify that every returned source belongs only to the selected document.

Retrieval is isolated through the selected document's `document_id` metadata filter.

## Evaluation

Place the local evaluation PDF under `data/raw/`, then run the single-document evaluation:

```bash
python -m scripts.evaluate_rag --pdf-path data/raw/KPC_2026.05.pdf --eval-path eval/rag_eval_questions.json --start-page 1 --end-page 3 --top-k 3 --use-cache
```

The following multi-document evaluation command is planned, but its script and dataset are not included yet:

```bash
python -m scripts.evaluate_multi_document_rag --eval-path eval/multi_document_eval.json --top-k 3 --relevance-mode strict
```

Use the manual validation flow above until the dedicated multi-document evaluator is implemented.

## Troubleshooting

- If the backend is unreachable, start it with `uvicorn app.main:app --reload`.
- If Ollama is unreachable, open the Ollama application and confirm that it is running.
- If the model is missing, run `ollama pull gemma3:4b`.
- Re-uploading the same `document_id` deletes its old Chroma chunks before re-indexing the new chunks.
- If Streamlit is unavailable, use Swagger at `http://127.0.0.1:8000/docs` or the CLI tools under `scripts/`.
