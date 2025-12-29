from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from specscanner import SpecIndex, build_index, load_index, query_index, save_index

app = FastAPI(title="SpecScanner", version="0.1.0")
INDEX_PATH = Path("specscanner_index.pkl")


def ensure_index() -> SpecIndex:
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="No index found. Ingest documents first.")
    return load_index(INDEX_PATH)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_endpoint(files: list[UploadFile] = File(...)) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        docs = []
        for upload in files:
            path = Path(tmpdir) / upload.filename
            content = await upload.read()
            path.write_bytes(content)
            docs.append(path)
        index = build_index(docs)
        save_index(index, INDEX_PATH)
    return {"chunks": len(index.chunks)}


@app.get("/query")
async def query_endpoint(question: str, top_k: int = 3) -> dict:
    index = ensure_index()
    passages = query_index(index, question, top_k=top_k)
    return {"question": question, "answers": passages}

