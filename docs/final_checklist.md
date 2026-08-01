# Final Checklist

## Code Quality

- [x] pytest passes (`226 passed`)
- [x] No secrets committed
- [x] `.env` is ignored
- [x] `.env.example` is committed
- [x] Local PDFs under `data/raw` and `data/documents` are ignored
- [x] Embedding cache and Chroma runtime files are ignored
- [x] Pytest only collects tests from `tests/`

## Functional Checks

- [ ] CLI works with fake provider
- [ ] CLI works with Ollama provider
- [x] FastAPI `/health` is covered by an API test
- [x] FastAPI `/ask` works with fake provider in automated tests
- [ ] FastAPI /ask works with Ollama provider
- [x] Document upload, listing, and document QA routes pass automated tests
- [x] Streamlit module and API client pass smoke tests
- [ ] Streamlit end-to-end flow works with Ollama
- [ ] Retrieval evaluation script runs
- [ ] Evaluation metrics are reported

## Demo Commands

`pytest`

```bash
python -m pytest
```

`evaluate_rag`

```bash
python -m scripts.evaluate_rag --pdf-path data/raw/KPC_2026.05.pdf --eval-path eval/rag_eval_questions.json --start-page 1 --end-page 3 --top-k 3 --use-cache
```

`ask_pdf`

```bash
python -m scripts.ask_pdf --question "Fonun yonetim ucreti nedir?" --llm-provider ollama --llm-model gemma3:4b
```

`uvicorn`

```bash
uvicorn app.main:app --reload
```

`streamlit`

```bash
streamlit run ui/app.py
```

`/ask` request sample

```powershell
$body = @{
    question = "Fonun yonetim ucreti nedir?"
    pdf_path = "data/raw/KPC_2026.05.pdf"
    start_page = 1
    end_page = 3
    top_k = 3
    answer_top_k = 2
    llm_provider = "ollama"
    llm_model = "gemma3:4b"
} | ConvertTo-Json -Depth 3

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/ask" `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

## Final Status

Final project status: Code-complete. Manual Ollama, CLI, and evaluation demo checks remain.
