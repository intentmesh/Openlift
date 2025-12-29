from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
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
DEFAULT_ISSUES_PATH = Path(__file__).resolve().parent / "issues.json"

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


@dataclass
class AnalyzerOptions:
    max_peaks: int = 5
    units: str = "mps2"
    highpass: float = 0.3
    lowpass: float = 40.0
    min_duration: float = 5.0
    min_sample_rate: float = 30.0
    issues_path: Path = DEFAULT_ISSUES_PATH
    stdout_report: bool = False
    enable_segments: bool = True
    segment_threshold: float = 0.2
    segment_min_duration: float = 2.0
    cloud_dir: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze elevator ride vibration data.")
    parser.add_argument("csv", type=Path, help="CSV file with timestamp, ax, ay, az columns.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for reports.")
    parser.add_argument("--plot", action="store_true", help="Generate spectrum plot (requires matplotlib).")
    parser.add_argument("--max-peaks", type=int, default=5, help="Maximum dominant peaks to report.")
    parser.add_argument("--units", choices=("mps2", "g"), default="mps2", help="Acceleration units in CSV.")
    parser.add_argument("--highpass", type=float, default=0.3, help="High-pass filter cutoff in Hz.")
    parser.add_argument("--lowpass", type=float, default=40.0, help="Low-pass filter cutoff in Hz.")
    parser.add_argument("--min-duration", type=float, default=5.0, help="Minimum recording length in seconds.")
    parser.add_argument("--min-sample-rate", type=float, default=30.0, help="Minimum sample rate in Hz.")
    parser.add_argument("--issues-config", type=Path, default=DEFAULT_ISSUES_PATH, help="Issue database JSON.")
    parser.add_argument("--stdout-report", action="store_true", help="Print the JSON payload to stdout.")
    parser.add_argument("--segment-threshold", type=float, default=0.2, help="Threshold for ride segments.")
    parser.add_argument("--segment-min-duration", type=float, default=2.0, help="Minimum segment duration (s).")
    parser.add_argument("--disable-segments", action="store_true", help="Skip ride segmentation analysis.")
    parser.add_argument("--cloud-dir", type=Path, default=None, help="Directory to mirror JSON payloads.")
    return parser.parse_args()


def build_options(args: argparse.Namespace) -> AnalyzerOptions:
    return AnalyzerOptions(
        max_peaks=args.max_peaks,
        units=args.units,
        highpass=args.highpass,
        lowpass=args.lowpass,
        min_duration=args.min_duration,
        min_sample_rate=args.min_sample_rate,
        issues_path=args.issues_config,
        stdout_report=args.stdout_report,
        enable_segments=not args.disable_segments,
        segment_threshold=args.segment_threshold,
        segment_min_duration=args.segment_min_duration,
        cloud_dir=args.cloud_dir,
    )


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"timestamp", "ax", "ay", "az"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {', '.join(sorted(missing))}")
    return df.sort_values("timestamp").reset_index(drop=True)


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
    return float(1.0 / np.median(dt))


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
    spectra["mag"] = np.sqrt(spectra["ax"] ** 2 + spectra["ay"] ** 2 + spectra["az"] ** 2)
    return freqs, spectra


def load_issue_db(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Issue DB not found at {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


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


def segment_dataframe(
    df: pd.DataFrame,
    sample_rate: float,
    threshold: float = 0.2,
    min_duration: float = 2.0,
) -> List[Dict[str, float]]:
    vector = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2).to_numpy(dtype=float)
    deviation = np.abs(vector - np.mean(vector))
    active = deviation > threshold
    segments: List[Dict[str, float]] = []
    start_idx = None

    def record_segment(start: int, end: int) -> None:
        duration = (end - start) / sample_rate
        if duration < min_duration:
            return
        seg_slice = vector[start:end]
        rms = float(np.sqrt(np.mean(seg_slice**2)))
        segments.append(
            {
                "start_time": float(df["timestamp"].iloc[start]),
                "end_time": float(df["timestamp"].iloc[end - 1]),
                "duration": duration,
                "rms_ms2": rms,
            }
        )

    for idx, is_active in enumerate(active):
        if is_active and start_idx is None:
            start_idx = idx
        elif not is_active and start_idx is not None:
            record_segment(start_idx, idx)
            start_idx = None

    if start_idx is not None:
        record_segment(start_idx, len(active))
    return segments


def export_to_cloud(payload: Dict[str, object], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    export_path = directory / f"openvibe_{timestamp}.json"
    export_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_reports(
    output_dir: Path,
    peaks: List[Peak],
    sample_rate: float,
    duration: float,
    metrics: IsoMetrics,
    segments: List[Dict[str, float]],
    plot_path: Path | None,
    persist: bool = True,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "sample_rate_hz": sample_rate,
        "duration_seconds": duration,
        "iso_metrics": asdict(metrics),
        "peaks": [asdict(peak) for peak in peaks],
        "segments": segments,
    }
    if not persist:
        return payload

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
    for peak in peaks:
        md_lines.append(
            f"| {peak.frequency:.2f} | {peak.amplitude:.2f} | {peak.issue} | {peak.recommendation} |"
        )

    md_lines.extend(["", "## Ride Segments"])
    if segments:
        md_lines.append("| Start (s) | End (s) | Duration (s) | RMS (m/s²) |")
        md_lines.append("| --- | --- | --- | --- |")
        for seg in segments:
            md_lines.append(
                f"| {seg['start_time']:.2f} | {seg['end_time']:.2f} | {seg['duration']:.2f} | {seg['rms_ms2']:.3f} |"
            )
    else:
        md_lines.append("No ride segments detected based on configured threshold.")

    if plot_path:
        rel_plot = plot_path.name
        md_lines.extend(["", f"![Spectrum plot]({rel_plot})"])

    md_lines.append("")
    md_lines.append("_Generated by OpenVibe – experimental diagnostic utility. Always confirm findings with certified instrumentation._")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


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


def analyze_dataframe(
    df: pd.DataFrame,
    options: AnalyzerOptions,
    issues: Sequence[Dict[str, object]],
    output_dir: Path,
    include_plot: bool = False,
    persist_reports: bool = True,
) -> Dict[str, object]:
    df = convert_units(df, options.units)
    sample_rate = estimate_sample_rate(df)
    duration = float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])
    validate_recording(duration, sample_rate, options.min_duration, options.min_sample_rate)
    filtered = filter_axes(df, sample_rate, options.highpass, options.lowpass)
    metrics = compute_iso_metrics(filtered, sample_rate)
    freqs, spectra = compute_fft_axes(filtered, sample_rate)
    peaks = detect_peaks(freqs, spectra["mag"], options.max_peaks, issues)
    segments = (
        segment_dataframe(filtered, sample_rate, options.segment_threshold, options.segment_min_duration)
        if options.enable_segments
        else []
    )
    plot_path = maybe_plot(freqs, spectra["mag"], output_dir) if (include_plot and persist_reports) else None
    payload = write_reports(output_dir, peaks, sample_rate, duration, metrics, segments, plot_path, persist_reports)
    if options.cloud_dir:
        export_to_cloud(payload, options.cloud_dir)
    return payload


def analyze_file(
    csv_path: Path,
    options: AnalyzerOptions,
    output_dir: Path | None = None,
    include_plot: bool = False,
    persist_reports: bool = True,
) -> Dict[str, object]:
    df = load_data(csv_path)
    issues = load_issue_db(options.issues_path)
    return analyze_dataframe(
        df,
        options,
        issues,
        output_dir or csv_path.parent,
        include_plot=include_plot,
        persist_reports=persist_reports,
    )


def main() -> None:
    args = parse_args()
    options = build_options(args)
    df = load_data(args.csv)
    output_dir = args.output or args.csv.parent
    issues = load_issue_db(options.issues_path)
    payload = analyze_dataframe(
        df,
        options,
        issues,
        output_dir,
        include_plot=args.plot,
        persist_reports=True,
    )
    if options.stdout_report:
        print(json.dumps(payload, indent=2))
    print(f"Report written to {output_dir.resolve()}")
    if not payload["peaks"]:
        print("No significant vibration peaks detected. Consider recording a longer trace.")


if __name__ == "__main__":
    main()

