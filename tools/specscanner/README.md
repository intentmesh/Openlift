# SpecScanner – Manual Ingestion & Q/A

SpecScanner lets mechanics ingest PDF/TXT manuals and ask natural-language questions against them using TF‑IDF retrieval.

## Quick Start

```bash
cd tools/specscanner
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ingest sample text
python specscanner.py ingest index.pkl samples/manual_excerpt.txt

# ask a question
python specscanner.py query index.pkl "What resets the drive fault?"
```

## REST API

```bash
uvicorn service:app --reload
# POST /ingest with files[] (PDF/TXT)
# GET  /query?question=...&top_k=3
```

## Directory Layout

- `specscanner.py` – CLI for ingest/query
- `service.py` – FastAPI wrapper
- `samples/manual_excerpt.txt` – demo content
- `tests/` – pytest coverage

## Roadmap
- Replace TF‑IDF with embeddings (sentence-transformers/FAISS)
- Add PDF OCR fallback
- Persist metadata (vendor, controller model)

