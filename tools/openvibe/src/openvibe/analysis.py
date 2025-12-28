from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ISSUE_DB: list[tuple[float, float, str, str]] = [
    (0.5, 2.0, "Counterweight balance", "Check car load balance and counterweight tension."),
    (
        2.0,
        5.0,
        "Door operator resonance",
        "Inspect door rollers, hanger alignment, and sill guides.",
    ),
    (
        5.0,
        9.0,
        "Rope piston / hoist sway",
        "Inspect rope tension, compensation chain, and damping pads.",
    ),
    (9.0, 14.0, "Guide roller wear", "Check guide shoes, roller bearings, and lubrication."),
    (
        14.0,
        25.0,
        "Drive/sheave alignment",
        "Inspect machine bedplate, motor bearings, and sheave wear.",
    ),
]


@dataclass(frozen=True)
class Peak:
    frequency: float
    amplitude: float
    issue: str
    recommendation: str


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"timestamp", "ax", "ay", "az"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {', '.join(sorted(missing))}")
    return df


def estimate_sample_rate(df: pd.DataFrame) -> float:
    timestamps = df["timestamp"].to_numpy(dtype=float)
    dt = np.diff(timestamps)
    dt = dt[dt > 0]
    median_dt = float(np.median(dt)) if dt.size else 0.0
    return (1.0 / median_dt) if median_dt > 0 else 50.0


def compute_fft(df: pd.DataFrame, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    # FFT the (demeaned) axes individually and combine magnitudes.
    #
    # Rationale: using the scalar magnitude sqrt(ax^2+ay^2+az^2) introduces a non-linearity
    # that can "rectify" simple sinusoidal signals (e.g., single-axis sine -> dominant 2*f).
    ax = df["ax"].to_numpy(dtype=float)
    ay = df["ay"].to_numpy(dtype=float)
    az = df["az"].to_numpy(dtype=float)
    if ax.size == 0:
        return np.array([]), np.array([])

    window = np.hanning(ax.size)
    x_fft = np.fft.rfft((ax - float(np.mean(ax))) * window)
    y_fft = np.fft.rfft((ay - float(np.mean(ay))) * window)
    z_fft = np.fft.rfft((az - float(np.mean(az))) * window)

    freqs = np.fft.rfftfreq(ax.size, d=1.0 / sample_rate)
    amplitude = np.sqrt(np.abs(x_fft) ** 2 + np.abs(y_fft) ** 2 + np.abs(z_fft) ** 2)
    return freqs, amplitude


def classify_peak(freq: float) -> tuple[str, str]:
    for f_low, f_high, issue, recommendation in ISSUE_DB:
        if f_low <= freq <= f_high:
            return issue, recommendation
    return "Unclassified vibration", "Collect more data and compare with baseline elevators."


def find_peaks(freqs: np.ndarray, amplitude: np.ndarray, max_peaks: int) -> list[Peak]:
    valid = (freqs >= 0.5) & (freqs <= 30.0)
    freqs = freqs[valid]
    amplitude = amplitude[valid]
    if freqs.size == 0:
        return []

    max_amp = float(np.max(amplitude)) if amplitude.size else 0.0
    norm_amp = (amplitude / max_amp) if max_amp > 0 else amplitude

    idx = np.argsort(norm_amp)[::-1][: max_peaks * 3]  # oversample then deduplicate close peaks
    peaks: list[Peak] = []
    seen_bins: set[float] = set()
    for i in idx:
        bin_key = round(float(freqs[i]) * 2) / 2  # bucket by 0.5 Hz to avoid duplicates
        if bin_key in seen_bins:
            continue
        seen_bins.add(bin_key)
        issue, rec = classify_peak(float(freqs[i]))
        peaks.append(Peak(float(freqs[i]), float(norm_amp[i]), issue, rec))
        if len(peaks) >= max_peaks:
            break
    return peaks


def write_reports(
    output_dir: Path,
    peaks: list[Peak],
    sample_rate: float,
    duration: float,
    plot_path: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "report.md"
    json_path = output_dir / "report.json"

    md_lines = [
        "# OpenVibe Ride Quality Report",
        "",
        f"- Sample rate: **{sample_rate:.2f} Hz**",
        f"- Recording duration: **{duration:.1f} s**",
        f"- Number of peaks: **{len(peaks)}**",
        "",
        "## Dominant Vibrations",
        "",
        "| Frequency (Hz) | Relative amplitude | Suspected issue | Recommendation |",
        "| --- | --- | --- | --- |",
    ]
    json_payload: dict[str, object] = {
        "sample_rate_hz": sample_rate,
        "duration_seconds": duration,
        "peaks": [],
    }

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

    if plot_path:
        md_lines.extend(["", f"![Spectrum plot]({plot_path.name})"])

    md_lines.append("")
    md_lines.append("_Generated by OpenVibe_")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


def maybe_plot(freqs: np.ndarray, amplitude: np.ndarray, output_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None

    plot_path = output_dir / "spectrum.png"
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, amplitude)
    plt.xlim(0, 30)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Vibration Spectrum")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path
