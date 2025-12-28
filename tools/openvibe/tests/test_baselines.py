from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from openvibe.baselines import add_fingerprint, get_or_build_model, score_features
from openvibe.fingerprint import build_fingerprint, write_fingerprint


def _sine_df(
    freq_hz: float, sample_rate_hz: float, seconds: float, amplitude: float
) -> pd.DataFrame:
    n = int(sample_rate_hz * seconds)
    t = np.arange(n, dtype=float) / sample_rate_hz
    ax = amplitude * np.sin(2 * math.pi * freq_hz * t)
    return pd.DataFrame(
        {"timestamp": t, "ax": ax, "ay": np.zeros_like(ax), "az": np.zeros_like(ax)}
    )


def test_baseline_build_and_score(tmp_path: Path) -> None:
    store = tmp_path / "baselines.json"
    tags = {"elevator": "E12"}

    # Baseline samples: amplitude 1.0
    for i in range(5):
        df = _sine_df(freq_hz=3.0, sample_rate_hz=100.0, seconds=4.0, amplitude=1.0)
        csv_path = tmp_path / f"base_{i}.csv"
        df.to_csv(csv_path, index=False)
        fp = build_fingerprint(df, input_csv=csv_path, units="m/s2", max_peaks=3)
        fp_path = tmp_path / f"base_{i}.fingerprint.json"
        write_fingerprint(fp, fp_path)
        add_fingerprint(store_path=store, tags=tags, fingerprint=json.loads(fp_path.read_text()))

    model = get_or_build_model(store_path=store, tags=tags)
    assert model.n == 5

    # Score a higher-amplitude run; should be noticeably non-zero.
    df2 = _sine_df(freq_hz=3.0, sample_rate_hz=100.0, seconds=4.0, amplitude=2.0)
    csv2 = tmp_path / "today.csv"
    df2.to_csv(csv2, index=False)
    fp2 = build_fingerprint(df2, input_csv=csv2, units="m/s2", max_peaks=3)

    score, rms_z, n_used, top = score_features(features=fp2.features, model=model)
    assert score > 5.0
    assert rms_z > 0
    assert n_used > 0
    assert top
