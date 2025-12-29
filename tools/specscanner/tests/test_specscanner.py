from pathlib import Path

import pytest

from specscanner import SpecIndex, build_index, query_index, save_index, load_index


def test_build_and_query(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_text("Door override relay closes after 10 seconds to reopen doors.", encoding="utf-8")
    index = build_index([sample])
    assert isinstance(index, SpecIndex)
    answers = query_index(index, "How long before door override?", top_k=1)
    assert answers and "10 seconds" in answers[0]


def test_persist_index(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_text("Drive reset button clears the fault.", encoding="utf-8")
    index = build_index([sample])
    idx_path = tmp_path / "index.pkl"
    save_index(index, idx_path)
    loaded = load_index(idx_path)
    answers = query_index(loaded, "How to clear drive fault?", top_k=1)
    assert answers

