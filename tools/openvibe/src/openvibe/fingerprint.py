from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from openvibe import __version__
from openvibe.reporting import analyze_df

FINGERPRINT_SCHEMA_VERSION = "openvibe.fingerprint.v1"


def stable_run_id(path: Path) -> str:
    resolved = str(path.resolve())
    return f"{path.stem}_{hashlib.sha1(resolved.encode('utf-8')).hexdigest()[:8]}"


def _band_key(low_hz: float, high_hz: float) -> str:
    def _fmt(x: float) -> str:
        # keep stable keys without dots
        s = f"{x:g}"
        return s.replace(".", "p")

    return f"band_{_fmt(low_hz)}_{_fmt(high_hz)}_fraction"


@dataclass(frozen=True)
class Fingerprint:
    schema_version: str
    tool_version: str
    run_id: str
    input_csv: str
    units: str
    features: dict[str, float]
    time_metrics: dict[str, float | str]
    spectral_bands: list[dict[str, float | str]]
    data_quality: dict[str, object]

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tool_version": self.tool_version,
            "run_id": self.run_id,
            "input": {"csv": self.input_csv, "units": self.units},
            "features": self.features,
            "time_metrics": self.time_metrics,
            "spectral_bands": self.spectral_bands,
            "data_quality": self.data_quality,
        }


def build_fingerprint(
    df: pd.DataFrame,
    *,
    input_csv: Path,
    units: str,
    max_peaks: int = 4,
) -> Fingerprint:
    run_id = stable_run_id(input_csv)
    sample_rate, duration, peaks, metrics, bands, load_stats = analyze_df(
        df, max_peaks=max_peaks, units=units
    )

    # Fixed-width feature vector for baselines/scoring.
    features: dict[str, float] = {
        "sample_rate_hz": float(sample_rate),
        "duration_seconds": float(duration),
        "accel_rms_vector_mps2": float(metrics.accel_rms_vector),
        "accel_p95_vector_mps2": float(metrics.accel_p95_vector),
        "jerk_rms_vector_mps3": float(metrics.jerk_rms_vector),
        "jerk_p95_vector_mps3": float(metrics.jerk_p95_vector),
        "gap_count": float(load_stats.get("gap_count", 0) or 0),
        "dt_median_s": float(load_stats.get("dt_median_s", 0.0) or 0.0),
        "dt_p95_s": float(load_stats.get("dt_p95_s", 0.0) or 0.0),
    }

    for b in bands:
        low = float(b["low_hz"])
        high = float(b["high_hz"])
        features[_band_key(low, high)] = float(b["fraction"])

    # Also keep a human-friendly top signal.
    if peaks:
        features["top_peak_hz"] = float(peaks[0].frequency)
        features["top_peak_amp"] = float(peaks[0].amplitude)
    else:
        features["top_peak_hz"] = 0.0
        features["top_peak_amp"] = 0.0

    return Fingerprint(
        schema_version=FINGERPRINT_SCHEMA_VERSION,
        tool_version=__version__,
        run_id=run_id,
        input_csv=str(input_csv),
        units=units,
        features=features,
        time_metrics=metrics.to_json(),
        spectral_bands=bands,
        data_quality=load_stats,
    )


def write_fingerprint(fp: Fingerprint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fp.to_json(), indent=2), encoding="utf-8")


def read_fingerprint(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
