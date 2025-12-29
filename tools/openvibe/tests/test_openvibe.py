from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
for candidate in (PACKAGE_ROOT, REPO_ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from openvibe import (  # noqa: E402
    AnalyzerOptions,
    IsoMetrics,
    analyze_dataframe,
    convert_units,
    detect_peaks,
    load_issue_db,
    compute_iso_metrics,
    validate_recording,
)
from openvibe.core import segment_dataframe

FIXTURE_DIR = PACKAGE_ROOT


def make_df(value: float = 1.0, length: int = 100, sample_rate: float = 50.0) -> pd.DataFrame:
    timestamps = np.arange(length, dtype=float) / sample_rate
    data = {
        "timestamp": timestamps,
        "ax": np.full(length, value),
        "ay": np.full(length, value),
        "az": np.full(length, 9.81 + value),
    }
    return pd.DataFrame(data)


def test_convert_units_from_g():
    df = make_df(value=0.1)
    converted = convert_units(df.copy(), "g")
    assert pytest.approx(converted["ax"].iloc[0], rel=1e-3) == 0.1 * 9.80665


def test_validate_recording_rejects_short_duration():
    with pytest.raises(ValueError):
        validate_recording(duration=2.0, sample_rate=60, min_duration=5.0, min_sample_rate=30.0)


def test_compute_iso_metrics_returns_expected_class():
    df = make_df(value=0.05)
    metrics = compute_iso_metrics(df, sample_rate=50.0)
    assert isinstance(metrics, IsoMetrics)
    assert metrics.comfort_class in {"A", "B", "C", "D"}


def test_detect_peaks_classifies_frequency(tmp_path: Path):
    freqs = np.linspace(0, 30, 601)
    amplitude = np.zeros_like(freqs)
    idx = np.argmin(np.abs(freqs - 10.0))
    amplitude[idx] = 1.0
    issues = load_issue_db(FIXTURE_DIR / "issues.json")
    peaks = detect_peaks(freqs, amplitude, max_peaks=3, issues=issues)
    assert peaks
    assert pytest.approx(peaks[0].frequency, abs=0.1) == 10.0
    assert "Guide" in peaks[0].issue


def test_segment_dataframe_detects_motion():
    sample_rate = 50.0
    length = 400
    timestamps = np.arange(length) / sample_rate
    variation = 0.5 * np.sin(2 * np.pi * 0.5 * timestamps)
    data = {
        "timestamp": timestamps,
        "ax": np.zeros(length),
        "ay": np.zeros(length),
        "az": 9.81 + variation,
    }
    df = pd.DataFrame(data)
    segments = segment_dataframe(df, sample_rate=50.0, threshold=0.05, min_duration=0.5)
    assert segments
    assert segments[0]["duration"] >= 0.5


def test_analyze_dataframe_payload(tmp_path: Path):
    df = make_df(value=0.2, length=400)
    options = AnalyzerOptions()
    issues = load_issue_db(FIXTURE_DIR / "issues.json")
    payload = analyze_dataframe(df, options, issues, tmp_path, persist_reports=False)
    assert "peaks" in payload and "segments" in payload

