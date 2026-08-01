# Final Checklist

## Tests

- [x] Full pytest suite passes
- [x] Single-document retrieval evaluation passes
- [ ] Multi-document evaluation passes
- [x] FastAPI smoke test passes
- [x] Streamlit smoke test passes
- [x] Ollama answer generation works

## Product Flow

- [x] PDF upload works
- [x] PDF is stored under document storage
- [x] Registry record is created
- [x] Chroma index is created
- [x] Multiple documents can be listed
- [x] Selected document can be queried
- [x] Sources belong only to the selected document
- [x] Re-indexing does not leave stale chunks

## Repository Hygiene

- [x] `.env` is ignored
- [x] `.env.example` is committed
- [x] Uploaded PDFs are ignored
- [x] Chroma runtime files are ignored
- [x] Document registry runtime file is ignored
- [x] No secrets are committed
- [x] README commands are current

## Documentation

- [x] README is final
- [x] Product usage document exists
- [x] Limitations are documented
- [x] Verified evaluation metrics are documented

## Final Status

```text
Project status: Ready for regular use and GitHub sharing.
```

This status should be confirmed only after all critical checks are complete.

Validation note: Core product checks passed on 2026-08-01. The dedicated
multi-document evaluation script and dataset remain planned; document isolation
is currently covered by automated metadata-filter tests and manual API checks.
