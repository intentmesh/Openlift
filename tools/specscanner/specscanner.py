#!/usr/bin/env python3
"""SpecScanner CLI – ingest PDF/TXT manuals and perform semantic search."""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SpecIndex:
    vectorizer: TfidfVectorizer
    matrix: np.ndarray
    chunks: List[str]


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_text(str(path))
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = 900) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buffer = ""
    for para in paragraphs:
        candidate = f"{buffer} {para}".strip()
        if len(candidate) >= chunk_size and buffer:
            chunks.append(buffer.strip())
            buffer = para
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer.strip())
    return chunks


def build_index(docs: List[Path]) -> SpecIndex:
    all_chunks: List[str] = []
    for doc in docs:
        text = read_document(doc)
        chunks = chunk_text(text)
        if not chunks:
            continue
        all_chunks.extend(chunks)
    if not all_chunks:
        raise ValueError("No content extracted from provided documents.")
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(all_chunks)
    return SpecIndex(vectorizer=vectorizer, matrix=matrix, chunks=all_chunks)


def save_index(index: SpecIndex, path: Path) -> None:
    with path.open("wb") as fh:
        pickle.dump(index, fh)


def load_index(path: Path) -> SpecIndex:
    with path.open("rb") as fh:
        return pickle.load(fh)


def query_index(index: SpecIndex, question: str, top_k: int = 3) -> List[str]:
    query_vec = index.vectorizer.transform([question])
    sims = cosine_similarity(query_vec, index.matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [index.chunks[i] for i in top_idx if sims[i] > 0]


def cli_ingest(args: argparse.Namespace) -> None:
    docs = [Path(p) for p in args.documents]
    index = build_index(docs)
    save_index(index, Path(args.index))
    print(f"Ingested {len(index.chunks)} chunks into {args.index}")


def cli_query(args: argparse.Namespace) -> None:
    index = load_index(Path(args.index))
    answers = query_index(index, args.question, top_k=args.top_k)
    if not answers:
        print("No relevant passages found.")
    else:
        for i, passage in enumerate(answers, 1):
            print(f"---- Match #{i} ----\n{passage}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpecScanner manual ingestion/search tool.")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest PDF/TXT manuals into an index.")
    ingest.add_argument("index", help="Output path for the index (pickle).")
    ingest.add_argument("documents", nargs="+", help="List of documents to ingest.")
    ingest.set_defaults(func=cli_ingest)

    query = sub.add_parser("query", help="Query an existing index.")
    query.add_argument("index", help="Path to existing index pickle.")
    query.add_argument("question", help="Question/prompt to search for.")
    query.add_argument("--top-k", type=int, default=3, help="Number of passages to return.")
    query.set_defaults(func=cli_query)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

