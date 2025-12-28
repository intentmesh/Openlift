from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from openvibe.analysis import compute_fft, estimate_sample_rate, find_peaks, load_data


def _sine_df(freq_hz: float, sample_rate_hz: float, seconds: float) -> pd.DataFrame:
    n = int(sample_rate_hz * seconds)
    t = np.arange(n, dtype=float) / sample_rate_hz
    ax = np.sin(2 * math.pi * freq_hz * t)
    return pd.DataFrame(
        {
            "timestamp": t,
            "ax": ax,
            "ay": np.zeros_like(ax),
            "az": np.zeros_like(ax),
        }
    )


def test_load_data_validates_columns(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": [0.0], "ax": [0.0]}).to_csv(p, index=False)
    try:
        load_data(p)
        raise AssertionError("Expected missing-column error")
    except ValueError as e:
        assert "Missing columns" in str(e)


def test_load_data_sorts_and_dedupes_timestamps(tmp_path: Path) -> None:
    p = tmp_path / "messy.csv"
    pd.DataFrame(
        {
            "timestamp": [2.0, 1.0, 1.0, "bad"],
            "ax": [0.0, 0.0, 0.0, 0.0],
            "ay": [0.0, 0.0, 0.0, 0.0],
            "az": [0.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(p, index=False)
    df = load_data(p)
    assert df["timestamp"].tolist() == [1.0, 2.0]


def test_estimate_sample_rate_matches_input() -> None:
    df = _sine_df(freq_hz=3.0, sample_rate_hz=100.0, seconds=2.0)
    sr = estimate_sample_rate(df)
    assert 95.0 <= sr <= 105.0


def test_find_peaks_detects_dominant_frequency() -> None:
    df = _sine_df(freq_hz=3.0, sample_rate_hz=100.0, seconds=4.0)
    sr = estimate_sample_rate(df)
    freqs, amp = compute_fft(df, sr)
    peaks = find_peaks(freqs, amp, max_peaks=3)
    assert peaks, "Expected at least one peak"
    assert abs(peaks[0].frequency - 3.0) < 0.5
