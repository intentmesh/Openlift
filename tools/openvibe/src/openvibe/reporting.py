from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from openvibe import __version__
from openvibe.analysis import (
    Peak,
    band_power_summary,
    compute_fft,
    estimate_sample_rate,
    find_peaks,
)
from openvibe.metrics import TimeMetrics, compute_time_metrics

SCHEMA_VERSION = "openvibe.report.v1"

DEFAULT_BANDS: list[tuple[float, float, str]] = [
    (0.5, 2.0, "0.5–2 Hz (Counterweight/balance)"),
    (2.0, 5.0, "2–5 Hz (Door resonance)"),
    (5.0, 9.0, "5–9 Hz (Hoist sway)"),
    (9.0, 14.0, "9–14 Hz (Guide rollers)"),
    (14.0, 25.0, "14–25 Hz (Drive/sheave)"),
    (25.0, 30.0, "25–30 Hz (Other)"),
]


@dataclass(frozen=True)
class PeakComparison:
    frequency_hz: float
    current_relative_amplitude: float
    baseline_relative_amplitude: float
    delta_relative_amplitude: float

    def to_json(self) -> dict[str, float]:
        return {
            "frequency_hz": self.frequency_hz,
            "current_relative_amplitude": self.current_relative_amplitude,
            "baseline_relative_amplitude": self.baseline_relative_amplitude,
            "delta_relative_amplitude": self.delta_relative_amplitude,
        }


def _bucket_hz(f: float) -> float:
    return round(f * 2) / 2


def compare_peaks(current: list[Peak], baseline: list[Peak]) -> list[PeakComparison]:
    """
    Compare two peak lists by 0.5 Hz buckets.
    """

    cur = {_bucket_hz(p.frequency): p for p in current}
    base = {_bucket_hz(p.frequency): p for p in baseline}
    keys = sorted(set(cur) | set(base))
    out: list[PeakComparison] = []
    for k in keys:
        c = cur.get(k)
        b = base.get(k)
        c_amp = c.amplitude if c else 0.0
        b_amp = b.amplitude if b else 0.0
        out.append(
            PeakComparison(
                frequency_hz=k,
                current_relative_amplitude=float(c_amp),
                baseline_relative_amplitude=float(b_amp),
                delta_relative_amplitude=float(c_amp - b_amp),
            )
        )
    return out


def _delta_metrics(current: TimeMetrics, baseline: TimeMetrics) -> dict[str, float]:
    # Both metrics are emitted in m/s2 / m/s3 already, so numeric deltas are meaningful.
    return {
        "accel_rms_vector": current.accel_rms_vector - baseline.accel_rms_vector,
        "jerk_rms_vector": current.jerk_rms_vector - baseline.jerk_rms_vector,
        "accel_p95_vector": current.accel_p95_vector - baseline.accel_p95_vector,
        "jerk_p95_vector": current.jerk_p95_vector - baseline.jerk_p95_vector,
    }


def _delta_spectral_bands(
    current: list[dict[str, float | str]],
    baseline: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    cur = {str(b["label"]): float(b["fraction"]) for b in current}
    base = {str(b["label"]): float(b["fraction"]) for b in baseline}
    labels = sorted(set(cur) | set(base))
    out: list[dict[str, float | str]] = []
    for label in labels:
        c = cur.get(label, 0.0)
        b = base.get(label, 0.0)
        out.append(
            {
                "label": label,
                "current_fraction": c,
                "baseline_fraction": b,
                "delta_fraction": c - b,
            }
        )
    return out


def analyze_df(
    df: pd.DataFrame, *, max_peaks: int, units: str
) -> tuple[
    float,
    float,
    list[Peak],
    TimeMetrics,
    list[dict[str, float | str]],
    dict[str, object],
]:
    load_stats = df.attrs.get("openvibe_load_stats", {})
    sample_rate = estimate_sample_rate(df)
    duration = float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) if len(df) else 0.0
    freqs, amplitude = compute_fft(df, sample_rate)
    peaks = find_peaks(freqs, amplitude, max_peaks)
    # Time-domain metrics are computed on the AC component by default (DC/gravity removed).
    metrics = compute_time_metrics(df, units=units, remove_dc=True)
    bands = band_power_summary(freqs, amplitude, bands=DEFAULT_BANDS)
    return sample_rate, duration, peaks, metrics, bands, dict(load_stats)


def write_reports(
    *,
    output_dir: Path,
    input_csv: Path,
    units: str,
    sample_rate_hz: float,
    duration_seconds: float,
    peaks: list[Peak],
    metrics: TimeMetrics,
    band_summary: list[dict[str, float | str]],
    load_stats: dict[str, object],
    plot_path: Path | None,
    baseline_csv: Path | None = None,
    baseline_peaks: list[Peak] | None = None,
    baseline_metrics: TimeMetrics | None = None,
    baseline_band_summary: list[dict[str, float | str]] | None = None,
    baseline_load_stats: dict[str, object] | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "report.md"
    json_path = output_dir / "report.json"

    md_lines: list[str] = [
        "# OpenVibe Ride Quality Report",
        "",
        f"- Tool version: **{__version__}**",
        f"- Input: **{input_csv.name}**",
        f"- Units: **{units}**",
        f"- Sample rate: **{sample_rate_hz:.2f} Hz**",
        f"- Recording duration: **{duration_seconds:.1f} s**",
        "",
        "## Data Quality",
        "",
        f"- Raw rows: **{int(load_stats.get('raw_rows', 0))}**",
        f"- Dropped rows (non-numeric/NA): **{int(load_stats.get('rows_dropped_non_numeric_or_na', 0))}**",
        f"- Dropped duplicate timestamps: **{int(load_stats.get('duplicate_timestamps_dropped', 0))}**",
        f"- Final rows: **{int(load_stats.get('final_rows', 0))}**",
        "",
        "## Time-Domain Metrics (m/s², m/s³)",
        "",
        f"- DC removed: **{metrics.dc_removed}**",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accel RMS (vector) | {metrics.accel_rms_vector:.4f} |",
        f"| Accel P95 (vector) | {metrics.accel_p95_vector:.4f} |",
        f"| Jerk RMS (vector) | {metrics.jerk_rms_vector:.4f} |",
        f"| Jerk P95 (vector) | {metrics.jerk_p95_vector:.4f} |",
        "",
        "## Spectral Band Energy (0.5–30 Hz)",
        "",
        "| Band | Power fraction |",
        "| --- | ---: |",
    ]

    json_payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": __version__,
        "input": {"csv": str(input_csv), "units": units},
        "data_quality": load_stats,
        "sample_rate_hz": sample_rate_hz,
        "duration_seconds": duration_seconds,
        "time_metrics": metrics.to_json(),
        "spectral_bands": band_summary,
        "peaks": [],
    }

    for b in band_summary:
        md_lines.append(f"| {b['label']} | {float(b['fraction']):.3f} |")

    md_lines.extend(
        [
            "",
            "## Dominant Vibrations (0.5–30 Hz)",
            "",
            "| Frequency (Hz) | Relative amplitude | Suspected issue | Recommendation |",
            "| ---: | ---: | --- | --- |",
        ]
    )

    for peak in peaks:
        md_lines.append(
            f"| {peak.frequency:.2f} | {peak.amplitude:.2f} | {peak.issue} | {peak.recommendation} |"
        )
        json_payload["peaks"].append(
            {
                "frequency_hz": peak.frequency,
                "relative_amplitude": peak.amplitude,
                "issue": peak.issue,
                "recommendation": peak.recommendation,
            }
        )

    if (
        baseline_csv
        and baseline_peaks is not None
        and baseline_metrics is not None
        and baseline_band_summary is not None
    ):
        delta_bands = _delta_spectral_bands(band_summary, baseline_band_summary)
        md_lines.extend(
            [
                "",
                "## Baseline Comparison",
                "",
                f"- Baseline input: **{baseline_csv.name}**",
                "",
            ]
        )
        if baseline_load_stats is not None:
            md_lines.extend(
                [
                    "### Baseline Data Quality",
                    "",
                    f"- Raw rows: **{int(baseline_load_stats.get('raw_rows', 0))}**",
                    f"- Dropped rows (non-numeric/NA): **{int(baseline_load_stats.get('rows_dropped_non_numeric_or_na', 0))}**",
                    f"- Dropped duplicate timestamps: **{int(baseline_load_stats.get('duplicate_timestamps_dropped', 0))}**",
                    f"- Final rows: **{int(baseline_load_stats.get('final_rows', 0))}**",
                    "",
                ]
            )
        md_lines.extend(
            [
                "### Delta Time Metrics (current − baseline)",
                "",
                "| Metric | Delta |",
                "| --- | ---: |",
                f"| Accel RMS (vector) | {_delta_metrics(metrics, baseline_metrics)['accel_rms_vector']:.4f} |",
                f"| Accel P95 (vector) | {_delta_metrics(metrics, baseline_metrics)['accel_p95_vector']:.4f} |",
                f"| Jerk RMS (vector) | {_delta_metrics(metrics, baseline_metrics)['jerk_rms_vector']:.4f} |",
                f"| Jerk P95 (vector) | {_delta_metrics(metrics, baseline_metrics)['jerk_p95_vector']:.4f} |",
                "",
                "### Delta Spectral Bands (fraction, current − baseline)",
                "",
                "| Band | Current | Baseline | Δ |",
                "| --- | ---: | ---: | ---: |",
            ]
        )

        for b in delta_bands:
            md_lines.append(
                f"| {b['label']} | {float(b['current_fraction']):.3f} |"
                f" {float(b['baseline_fraction']):.3f} | {float(b['delta_fraction']):+.3f} |"
            )

        md_lines.extend(
            [
                "",
                "### Peak Delta (bucketed at 0.5 Hz)",
                "",
                "| Frequency (Hz) | Current | Baseline | Δ |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )

        comps = compare_peaks(peaks, baseline_peaks)
        for c in comps:
            md_lines.append(
                f"| {c.frequency_hz:.2f} | {c.current_relative_amplitude:.2f} |"
                f" {c.baseline_relative_amplitude:.2f} | {c.delta_relative_amplitude:.2f} |"
            )

        json_payload["baseline"] = {
            "csv": str(baseline_csv),
            "data_quality": baseline_load_stats or {},
            "time_metrics": baseline_metrics.to_json(),
            "delta_time_metrics": _delta_metrics(metrics, baseline_metrics),
            "spectral_bands": baseline_band_summary,
            "delta_spectral_bands": delta_bands,
            "peak_deltas": [c.to_json() for c in comps],
        }

    if plot_path:
        md_lines.extend(["", f"![Spectrum plot]({plot_path.name})"])
        json_payload["plot"] = {"spectrum_png": str(plot_path)}

    md_lines.append("")
    md_lines.append("_Generated by OpenVibe_")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    return md_path, json_path, json_payload
