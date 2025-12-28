from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from openvibe.analysis import (
    band_power_summary,
    compute_fft,
    estimate_sample_rate,
    find_peaks,
    load_data,
)
from openvibe.reporting import DEFAULT_BANDS


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
    assert df["timestamp"].tolist() == [0.0, 1.0]
    stats = df.attrs.get("openvibe_load_stats", {})
    assert stats.get("raw_rows") == 4
    assert stats.get("final_rows") == 2


def test_load_data_stats_counts_dedupes(tmp_path: Path) -> None:
    p = tmp_path / "dupes.csv"
    pd.DataFrame(
        {
            "timestamp": [0.0, 0.0, 0.1],
            "ax": [0.0, 0.0, 0.0],
            "ay": [0.0, 0.0, 0.0],
            "az": [0.0, 0.0, 0.0],
        }
    ).to_csv(p, index=False)
    df = load_data(p)
    stats = df.attrs.get("openvibe_load_stats", {})
    assert stats.get("duplicate_timestamps_dropped") == 1


def test_load_data_column_aliases_and_semicolon_delimiter(tmp_path: Path) -> None:
    p = tmp_path / "aliases.csv"
    # Semicolon-delimited, mixed-case headers with units in names.
    p.write_text(
        "\n".join(
            [
                "Time;Accel X (m/s^2);Accel Y (m/s^2);Accel Z (m/s^2)",
                "0.00;0;0;0",
                "0.01;1;0;0",
                "0.02;0;0;0",
            ]
        ),
        encoding="utf-8",
    )
    df = load_data(p)
    assert list(df.columns) == ["timestamp", "ax", "ay", "az"]
    assert df.attrs["openvibe_load_stats"]["detected_columns"]["timestamp"] == "Time"


def test_timestamp_unit_auto_detects_milliseconds(tmp_path: Path) -> None:
    p = tmp_path / "ms.csv"
    pd.DataFrame(
        {
            "timestamp": [0, 10, 20, 30],  # ms
            "ax": [0, 0, 0, 0],
            "ay": [0, 0, 0, 0],
            "az": [0, 0, 0, 0],
        }
    ).to_csv(p, index=False)
    df = load_data(p)
    # Normalized to seconds and zeroed, so final timestamp should be 0.03s
    assert abs(float(df["timestamp"].iloc[-1]) - 0.03) < 1e-9
    stats = df.attrs.get("openvibe_load_stats", {})
    assert stats.get("timestamp_unit") == "ms"
    assert stats.get("dt_median_s") == 0.01


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


def test_band_power_summary_places_sine_in_expected_band() -> None:
    df = _sine_df(freq_hz=3.0, sample_rate_hz=100.0, seconds=4.0)
    sr = estimate_sample_rate(df)
    freqs, amp = compute_fft(df, sr)
    bands = band_power_summary(freqs, amp, bands=DEFAULT_BANDS)
    top = max(bands, key=lambda b: float(b["fraction"]))
    assert "2–5 Hz" in str(top["label"])
