from __future__ import annotations

import math

import numpy as np
import pandas as pd

from openvibe.metrics import G0_MPS2, compute_time_metrics


def _sine_df(
    freq_hz: float, sample_rate_hz: float, seconds: float, amplitude: float
) -> pd.DataFrame:
    n = int(sample_rate_hz * seconds)
    t = np.arange(n, dtype=float) / sample_rate_hz
    ax = amplitude * np.sin(2 * math.pi * freq_hz * t)
    return pd.DataFrame(
        {
            "timestamp": t,
            "ax": ax,
            "ay": np.zeros_like(ax),
            "az": np.zeros_like(ax),
        }
    )


def test_time_metrics_rms_for_sine_mps2() -> None:
    df = _sine_df(freq_hz=3.0, sample_rate_hz=500.0, seconds=4.0, amplitude=1.0)
    m = compute_time_metrics(df, units="m/s2")
    assert abs(m.accel_rms_vector - (1.0 / math.sqrt(2.0))) < 0.02

    expected_jerk_rms = (2.0 * math.pi * 3.0) / math.sqrt(2.0)
    assert abs(m.jerk_rms_vector - expected_jerk_rms) < 0.2


def test_time_metrics_units_g_scales_to_mps2() -> None:
    df = _sine_df(freq_hz=2.0, sample_rate_hz=500.0, seconds=4.0, amplitude=1.0)  # 1 g peak
    m = compute_time_metrics(df, units="g")
    assert abs(m.accel_rms_vector - (G0_MPS2 / math.sqrt(2.0))) < 0.05
