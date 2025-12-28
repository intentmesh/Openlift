#!/usr/bin/env python3
"""
OpenVibe - Elevator ride-quality analyzer.

Usage:
    python openvibe.py sample_data.csv --plot
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

G = 9.80665  # gravity constant
DEFAULT_ISSUES_PATH = Path(__file__).with_name("issues.json")

ISO_COMFORT = [
    (0.0, 0.315, "A", "Excellent ride comfort"),
    (0.315, 0.63, "B", "Good ride comfort"),
    (0.63, 1.0, "C", "Noticeable vibration"),
    (1.0, float("inf"), "D", "Uncomfortable vibration"),
]


@dataclass
class Peak:
    frequency: float
    amplitude: float
    issue: str
    recommendation: str


@dataclass
class IsoMetrics:
    rms_ms2: float
    jerk_rms_ms3: float
    comfort_class: str
    comfort_note: str


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
        default=5,
        help="Maximum dominant peaks to report (default: 5).",
    )
    parser.add_argument(
        "--units",
        choices=("mps2", "g"),
        default="mps2",
        help="Acceleration units in CSV (default: mps2).",
    )
    parser.add_argument(
        "--highpass",
        type=float,
        default=0.3,
        help="High-pass filter cutoff in Hz (default: 0.3, set 0 to disable).",
    )
    parser.add_argument(
        "--lowpass",
        type=float,
        default=40.0,
        help="Low-pass filter cutoff in Hz (default: 40.0, set 0 to disable).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=5.0,
        help="Minimum recording length in seconds (default: 5s).",
    )
    parser.add_argument(
        "--min-sample-rate",
        type=float,
        default=30.0,
        help="Minimum acceptable sample rate in Hz (default: 30Hz).",
    )
    parser.add_argument(
        "--issues-config",
        type=Path,
        default=DEFAULT_ISSUES_PATH,
        help="Path to vibration issue database (JSON).",
    )
    parser.add_argument(
        "--stdout-report",
        action="store_true",
        help="Print the JSON payload to stdout (useful for pipelines).",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"timestamp", "ax", "ay", "az"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {', '.join(sorted(missing))}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def convert_units(df: pd.DataFrame, units: str) -> pd.DataFrame:
    if units == "g":
        for axis in ("ax", "ay", "az"):
            df[axis] = df[axis] * G
    return df


def estimate_sample_rate(df: pd.DataFrame) -> float:
    timestamps = df["timestamp"].to_numpy(dtype=float)
    dt = np.diff(timestamps)
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("Timestamps must be strictly increasing.")
    median_dt = np.median(dt)
    return float(1.0 / median_dt)


def validate_recording(duration: float, sample_rate: float, min_duration: float, min_sample_rate: float) -> None:
    if duration < min_duration:
        raise ValueError(f"Recording too short ({duration:.1f}s). Collect at least {min_duration}s.")
    if sample_rate < min_sample_rate:
        raise ValueError(f"Sample rate {sample_rate:.1f}Hz is below required {min_sample_rate}Hz.")


def butter_filter(signal: np.ndarray, sample_rate: float, cut: float, btype: str) -> np.ndarray:
    if cut <= 0:
        return signal
    nyq = 0.5 * sample_rate
    norm = cut / nyq
    if norm >= 1:
        return signal
    b, a = sp_signal.butter(4, norm, btype=btype)
    return sp_signal.filtfilt(b, a, signal)


def filter_axes(df: pd.DataFrame, sample_rate: float, highpass: float, lowpass: float) -> pd.DataFrame:
    filtered = df.copy()
    for axis in ("ax", "ay", "az"):
        sig = filtered[axis].to_numpy(dtype=float)
        sig = butter_filter(sig, sample_rate, highpass, "highpass")
        sig = butter_filter(sig, sample_rate, lowpass, "lowpass")
        filtered[axis] = sig
    return filtered


def compute_iso_metrics(df: pd.DataFrame, sample_rate: float) -> IsoMetrics:
    vector = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2).to_numpy(dtype=float)
    rms = float(np.sqrt(np.mean(vector ** 2)))
    jerk = np.gradient(vector, 1.0 / sample_rate)
    jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))
    comfort_class, comfort_note = ISO_COMFORT[-1][2], ISO_COMFORT[-1][3]
    for low, high, cls, note in ISO_COMFORT:
        if low <= rms < high:
            comfort_class, comfort_note = cls, note
            break
    return IsoMetrics(rms, jerk_rms, comfort_class, comfort_note)


def compute_fft_axes(df: pd.DataFrame, sample_rate: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n = df.shape[0]
    window = np.hanning(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    spectra: Dict[str, np.ndarray] = {}
    for axis in ("ax", "ay", "az"):
        signal = df[axis].to_numpy(dtype=float) * window
        fft = np.fft.rfft(signal)
        spectra[axis] = np.abs(fft)
    vector_mag = np.sqrt(spectra["ax"] ** 2 + spectra["ay"] ** 2 + spectra["az"] ** 2)
    spectra["mag"] = vector_mag
    return freqs, spectra


def load_issue_db(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Issue DB not found at {path}")
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data


def classify_peak(freq: float, issues: Sequence[Dict[str, object]]) -> Tuple[str, str]:
    for entry in issues:
        f_low = float(entry["f_low"])
        f_high = float(entry["f_high"])
        if f_low <= freq <= f_high:
            return str(entry["issue"]), str(entry["recommendation"])
    return "Unclassified vibration", "Collect more data and compare with baseline elevators."


def detect_peaks(freqs: np.ndarray, amplitude: np.ndarray, max_peaks: int, issues: Sequence[Dict[str, object]]) -> List[Peak]:
    valid = (freqs >= 0.5) & (freqs <= 30.0)
    f_valid = freqs[valid]
    a_valid = amplitude[valid]
    if f_valid.size == 0 or np.max(a_valid) == 0:
        return []
    prominence = 0.08 * np.max(a_valid)
    peak_idx, _ = sp_signal.find_peaks(a_valid, prominence=prominence, distance=3)
    sorted_idx = peak_idx[np.argsort(a_valid[peak_idx])[::-1]]
    peaks: List[Peak] = []
    for idx in sorted_idx[:max_peaks]:
        freq = float(f_valid[idx])
        rel_amp = float(a_valid[idx] / np.max(a_valid))
        issue, rec = classify_peak(freq, issues)
        peaks.append(Peak(freq, rel_amp, issue, rec))
    return peaks


def write_reports(
    output_dir: Path,
    peaks: List[Peak],
    sample_rate: float,
    duration: float,
    metrics: IsoMetrics,
    plot_path: Path | None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "report.md"
    json_path = output_dir / "report.json"

    md_lines = [
        "# OpenVibe Ride Quality Report",
        "",
        f"- Sample rate: **{sample_rate:.2f} Hz**",
        f"- Recording duration: **{duration:.1f} s**",
        "",
        "## ISO 18738 Metrics",
        "",
        f"- RMS acceleration: **{metrics.rms_ms2:.3f} m/s²**",
        f"- RMS jerk: **{metrics.jerk_rms_ms3:.3f} m/s³**",
        f"- Comfort class: **{metrics.comfort_class}** ({metrics.comfort_note})",
        "",
        "## Dominant Vibrations",
        "",
        "| Frequency (Hz) | Relative amplitude | Suspected issue | Recommendation |",
        "| --- | --- | --- | --- |",
    ]
    json_payload: Dict[str, object] = {
        "sample_rate_hz": sample_rate,
        "duration_seconds": duration,
        "iso_metrics": asdict(metrics),
        "peaks": [],
    }

    for peak in peaks:
        md_lines.append(
            f"| {peak.frequency:.2f} | {peak.amplitude:.2f} | {peak.issue} | {peak.recommendation} |"
        )
        json_payload["peaks"].append(asdict(peak))

    if plot_path:
        rel_plot = plot_path.name
        md_lines.extend(["", f"![Spectrum plot]({rel_plot})"])

    md_lines.append("")
    md_lines.append("_Generated by OpenVibe – experimental diagnostic utility. Always confirm findings with certified instrumentation._")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    return json_payload


def maybe_plot(freqs: np.ndarray, amplitude: np.ndarray, output_dir: Path) -> Path | None:
    if plt is None:
        return None
    plot_path = output_dir / "spectrum.png"
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, amplitude)
    plt.xlim(0, 30)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Vibration Spectrum (Magnitude)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def main() -> None:
    args = parse_args()
    df = load_data(args.csv)
    df = convert_units(df, args.units)
    output_dir = args.output or args.csv.parent

    issues = load_issue_db(args.issues_config)
    sample_rate = estimate_sample_rate(df)
    duration = float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])
    validate_recording(duration, sample_rate, args.min_duration, args.min_sample_rate)

    filtered = filter_axes(df, sample_rate, args.highpass, args.lowpass)
    metrics = compute_iso_metrics(filtered, sample_rate)
    freqs, spectra = compute_fft_axes(filtered, sample_rate)
    peaks = detect_peaks(freqs, spectra["mag"], args.max_peaks, issues)
    plot_path = maybe_plot(freqs, spectra["mag"], output_dir) if args.plot else None
    payload = write_reports(output_dir, peaks, sample_rate, duration, metrics, plot_path)
    if args.stdout_report:
        print(json.dumps(payload, indent=2))
    print(f"Report written to {output_dir.resolve()}")
    if not peaks:
        print("No significant vibration peaks detected. Consider recording a longer trace.")


if __name__ == "__main__":
    main()



