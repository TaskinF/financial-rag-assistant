# Final Checklist

## Tests

- [ ] Full pytest suite passes
- [ ] Single-document retrieval evaluation passes
- [ ] Multi-document evaluation passes
- [ ] FastAPI smoke test passes
- [ ] Streamlit smoke test passes
- [ ] Ollama answer generation works

## Product Flow

- [ ] PDF upload works
- [ ] PDF is stored under document storage
- [ ] Registry record is created
- [ ] Chroma index is created
- [ ] Multiple documents can be listed
- [ ] Selected document can be queried
- [ ] Sources belong only to the selected document
- [ ] Re-indexing does not leave stale chunks

## Repository Hygiene

- [ ] `.env` is ignored
- [ ] `.env.example` is committed
- [ ] Uploaded PDFs are ignored
- [ ] Chroma runtime files are ignored
- [ ] Document registry runtime file is ignored
- [ ] No secrets are committed
- [ ] README commands are current

## Documentation

- [ ] README is final
- [ ] Product usage document exists
- [ ] Limitations are documented
- [ ] Verified evaluation metrics are documented

## Final Status

```text
Project status: Ready for regular use and GitHub sharing.
```

This status should be confirmed only after all critical checks are complete.
