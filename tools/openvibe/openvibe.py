#!/usr/bin/env python3
"""
OpenVibe - Elevator ride-quality analyzer.

Usage:
    python openvibe.py sample_data.csv --plot
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


ISSUE_DB: List[Tuple[float, float, str, str]] = [
    (0.5, 2.0, "Counterweight balance", "Check car load balance and counterweight tension."),
    (2.0, 5.0, "Door operator resonance", "Inspect door rollers, hanger alignment, and sill guides."),
    (5.0, 9.0, "Rope piston / hoist sway", "Inspect rope tension, compensation chain, and damping pads."),
    (9.0, 14.0, "Guide roller wear", "Check guide shoes, roller bearings, and lubrication."),
    (14.0, 25.0, "Drive/sheave alignment", "Inspect machine bedplate, motor bearings, and sheave wear."),
]


@dataclass
class Peak:
    frequency: float
    amplitude: float
    issue: str
    recommendation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze elevator ride vibration data.")
    parser.add_argument("csv", type=Path, help="CSV file with timestamp, ax, ay, az columns.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for reports (default: alongside CSV).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate spectrum plot (requires matplotlib).",
    )
    parser.add_argument(
        "--max-peaks",
        type=int,
        default=4,
        help="Number of dominant peaks to report (default: 4).",
    )
    return parser.parse_args()


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
    median_dt = np.median(dt[dt > 0])
    return float(1.0 / median_dt) if median_dt > 0 else 50.0


def compute_fft(df: pd.DataFrame, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    vector = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2).to_numpy(dtype=float)
    signal = vector - np.mean(vector)
    window = np.hanning(signal.size)
    windowed = signal * window
    fft = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    amplitude = np.abs(fft)
    return freqs, amplitude


def classify_peak(freq: float) -> Tuple[str, str]:
    for f_low, f_high, issue, recommendation in ISSUE_DB:
        if f_low <= freq <= f_high:
            return issue, recommendation
    return "Unclassified vibration", "Collect more data and compare with baseline elevators."


def find_peaks(freqs: np.ndarray, amplitude: np.ndarray, max_peaks: int) -> List[Peak]:
    valid = (freqs >= 0.5) & (freqs <= 30.0)
    freqs = freqs[valid]
    amplitude = amplitude[valid]
    if freqs.size == 0:
        return []

    norm_amp = amplitude / np.max(amplitude) if np.max(amplitude) > 0 else amplitude
    idx = np.argsort(norm_amp)[::-1][: max_peaks * 3]  # oversample then deduplicate close peaks
    peaks: List[Peak] = []
    seen_bins: set = set()
    for i in idx:
        bin_key = round(freqs[i] * 2) / 2  # bucket by 0.5 Hz to avoid duplicates
        if bin_key in seen_bins:
            continue
        seen_bins.add(bin_key)
        issue, rec = classify_peak(freqs[i])
        peaks.append(Peak(float(freqs[i]), float(norm_amp[i]), issue, rec))
        if len(peaks) >= max_peaks:
            break
    return peaks


def write_reports(
    output_dir: Path,
    peaks: List[Peak],
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
    json_payload: Dict[str, object] = {
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
        rel_plot = plot_path.name
        md_lines.extend(["", f"![Spectrum plot]({rel_plot})"])

    md_lines.append("")
    md_lines.append("_Generated by OpenVibe – tools/openvibe_")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


def maybe_plot(freqs: np.ndarray, amplitude: np.ndarray, output_dir: Path) -> Path | None:
    if plt is None:
        return None
    plot_path = output_dir / "spectrum.png"
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, amplitude)
    plt.xlim(0, 30)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Vibration Spectrum")
    plt.tight_grid = True
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def main() -> None:
    args = parse_args()
    df = load_data(args.csv)
    output_dir = args.output or args.csv.parent
    sample_rate = estimate_sample_rate(df)
    freqs, amplitude = compute_fft(df, sample_rate)
    peaks = find_peaks(freqs, amplitude, args.max_peaks)
    duration = float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])
    plot_path = maybe_plot(freqs, amplitude, output_dir) if args.plot else None
    write_reports(output_dir, peaks, sample_rate, duration, plot_path)
    print(f"Report written to {output_dir.resolve()}")
    if not peaks:
        print("No significant vibration peaks detected. Consider recording a longer trace.")


if __name__ == "__main__":
    main()

