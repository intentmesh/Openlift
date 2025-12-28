from __future__ import annotations

import json
import re
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


_ALIASES: dict[str, set[str]] = {
    "timestamp": {
        "timestamp",
        "time",
        "t",
        "ts",
        "epoch",
        "seconds",
        "sec",
        "datetime",
        "date_time",
    },
    "ax": {
        "ax",
        "accel_x",
        "accelx",
        "acc_x",
        "x",
        "x_accel",
        "x_acceleration",
        "linear_acceleration_x",
        "user_acceleration_x",
    },
    "ay": {
        "ay",
        "accel_y",
        "accely",
        "acc_y",
        "y",
        "y_accel",
        "y_acceleration",
        "linear_acceleration_y",
        "user_acceleration_y",
    },
    "az": {
        "az",
        "accel_z",
        "accelz",
        "acc_z",
        "z",
        "z_accel",
        "z_acceleration",
        "linear_acceleration_z",
        "user_acceleration_z",
    },
}


def _normalize_col(name: str) -> str:
    # Lowercase, convert punctuation/spaces to underscores, collapse repeats.
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _infer_column_map(columns: list[str]) -> dict[str, str]:
    normalized_to_original = {_normalize_col(c): c for c in columns}
    normalized = set(normalized_to_original.keys())

    out: dict[str, str] = {}
    for canonical, aliases in _ALIASES.items():
        if canonical in normalized:
            out[canonical] = normalized_to_original[canonical]
            continue
        # Exact alias match.
        hit = next((a for a in aliases if a in normalized), None)
        if hit is not None:
            out[canonical] = normalized_to_original[hit]
            continue
        # Fuzzy alias match (e.g., "accel_x_m_s_2" contains "accel_x").
        fuzzy = next(
            (
                n
                for n in normalized
                for a in aliases
                if n.startswith(f"{a}_") or f"_{a}_" in f"_{n}_" or n.endswith(f"_{a}")
            ),
            None,
        )
        if fuzzy is not None:
            out[canonical] = normalized_to_original[fuzzy]
    return out


def _infer_timestamp_unit_scale(
    timestamps: np.ndarray, dt: np.ndarray, timestamp_unit: str
) -> tuple[str, float]:
    """
    Return (unit, scale_to_seconds).

    timestamp_unit can be: auto, s, ms, us, ns
    """

    unit = timestamp_unit.lower().strip()
    if unit in {"s", "sec", "second", "seconds"}:
        return "s", 1.0
    if unit in {"ms", "milli", "milliseconds"}:
        return "ms", 1e-3
    if unit in {"us", "micro", "microseconds"}:
        return "us", 1e-6
    if unit in {"ns", "nano", "nanoseconds"}:
        return "ns", 1e-9
    if unit not in {"auto"}:
        raise ValueError("timestamp_unit must be one of: auto, s, ms, us, ns")

    if timestamps.size == 0:
        return "s", 1.0

    med_abs = float(np.median(np.abs(timestamps)))
    # Absolute Unix timestamp magnitude heuristic.
    if med_abs >= 1e17:
        return "ns", 1e-9
    if med_abs >= 1e14:
        return "us", 1e-6
    if med_abs >= 1e11:
        return "ms", 1e-3

    # Relative timestamps (often start at ~0). Use delta scale heuristic.
    pos_dt = dt[dt > 0]
    med_dt = float(np.median(pos_dt)) if pos_dt.size else 0.0
    if med_dt >= 1e6:
        return "ns", 1e-9
    if med_dt >= 1e3:
        return "us", 1e-6
    # Most phone logs are 25–200 Hz -> dt in ms is usually 5–40.
    if med_dt >= 10.0:
        return "ms", 1e-3
    return "s", 1.0


def load_data(path: Path, *, timestamp_unit: str = "auto") -> pd.DataFrame:
    # sep=None lets pandas sniff delimiters (comma, semicolon, tab, etc).
    df = pd.read_csv(path, sep=None, engine="python")
    column_map = _infer_column_map(list(df.columns))
    missing = [c for c in ["timestamp", "ax", "ay", "az"] if c not in column_map]
    if missing:
        raise ValueError(
            "Missing columns in CSV. "
            f"Need: timestamp, ax, ay, az. Missing: {', '.join(missing)}. "
            f"Found columns: {', '.join(map(str, df.columns))}"
        )

    raw_rows = int(len(df))

    # Select and rename to canonical column names.
    detected_columns = {k: v for k, v in column_map.items() if k in {"timestamp", "ax", "ay", "az"}}
    df = df[
        [
            detected_columns["timestamp"],
            detected_columns["ax"],
            detected_columns["ay"],
            detected_columns["az"],
        ]
    ]
    df = df.rename(
        columns={
            detected_columns["timestamp"]: "timestamp",
            detected_columns["ax"]: "ax",
            detected_columns["ay"]: "ay",
            detected_columns["az"]: "az",
        }
    )

    # Real-world mobile exports can contain blank rows, strings, NaNs, duplicate timestamps,
    # and out-of-order samples. Normalize to a clean, monotonic trace.
    timestamp_was_datetime = False
    raw_timestamp_series = df["timestamp"].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    # If timestamp couldn't be parsed as numeric, try datetime parsing.
    if float(df["timestamp"].isna().mean()) > 0.5:
        dt_parsed = pd.to_datetime(raw_timestamp_series, errors="coerce", utc=True)
        if float(dt_parsed.isna().mean()) < 0.5:
            timestamp_was_datetime = True
            df["timestamp"] = (dt_parsed.view("int64").astype(float)) / 1e9

    for col in ["ax", "ay", "az"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["timestamp", "ax", "ay", "az"]
    na_rows = int(df[required].isna().any(axis=1).sum())

    df = df.dropna(subset=required).copy()
    df = df.sort_values("timestamp", kind="mergesort")
    before_dedup = int(len(df))
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    duplicate_timestamps_dropped = before_dedup - int(len(df))
    df = df.reset_index(drop=True)

    raw_ts_min = float(df["timestamp"].iloc[0]) if len(df) else None
    raw_ts_max = float(df["timestamp"].iloc[-1]) if len(df) else None

    timestamps = df["timestamp"].to_numpy(dtype=float)
    dt = np.diff(timestamps)
    inferred_unit, scale = _infer_timestamp_unit_scale(timestamps, dt, timestamp_unit)
    if scale != 1.0 and len(df):
        df["timestamp"] = df["timestamp"] * scale

    # Normalize timebase to start at 0 (makes downstream metrics stable across devices).
    if len(df):
        t0 = float(df["timestamp"].iloc[0])
        df["timestamp"] = df["timestamp"] - t0

    ts_min = float(df["timestamp"].iloc[0]) if len(df) else None
    ts_max = float(df["timestamp"].iloc[-1]) if len(df) else None

    # Data integrity stats on normalized timestamps.
    ts = df["timestamp"].to_numpy(dtype=float)
    dt_norm = np.diff(ts)
    pos_dt_norm = dt_norm[dt_norm > 0]
    dt_median_s = float(np.median(pos_dt_norm)) if pos_dt_norm.size else None
    dt_p95_s = float(np.percentile(pos_dt_norm, 95.0)) if pos_dt_norm.size else None
    dt_min_s = float(np.min(pos_dt_norm)) if pos_dt_norm.size else None
    dt_max_s = float(np.max(pos_dt_norm)) if pos_dt_norm.size else None
    gap_threshold_s = (3.0 * dt_median_s) if dt_median_s else None
    gap_count = int(np.sum(pos_dt_norm > gap_threshold_s)) if gap_threshold_s else 0
    max_gap_s = float(np.max(pos_dt_norm[pos_dt_norm > gap_threshold_s])) if gap_count else None

    df.attrs["openvibe_load_stats"] = {
        "raw_rows": raw_rows,
        "rows_dropped_non_numeric_or_na": na_rows,
        "duplicate_timestamps_dropped": duplicate_timestamps_dropped,
        "final_rows": int(len(df)),
        "timestamp_unit": inferred_unit,
        "timestamp_was_datetime": timestamp_was_datetime,
        "timestamp_unit_override": timestamp_unit.lower().strip() != "auto",
        "raw_timestamp_min": raw_ts_min,
        "raw_timestamp_max": raw_ts_max,
        "timestamp_min": ts_min,
        "timestamp_max": ts_max,
        "dt_median_s": dt_median_s,
        "dt_p95_s": dt_p95_s,
        "dt_min_s": dt_min_s,
        "dt_max_s": dt_max_s,
        "gap_threshold_s": gap_threshold_s,
        "gap_count": gap_count,
        "max_gap_s": max_gap_s,
        "detected_columns": detected_columns,
    }
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


def band_power_summary(
    freqs: np.ndarray,
    amplitude: np.ndarray,
    *,
    bands: list[tuple[float, float, str]],
    min_hz: float = 0.5,
    max_hz: float = 30.0,
) -> list[dict[str, float | str]]:
    """
    Summarize spectral power by frequency bands.

    Uses power ~ amplitude^2 of the combined-axis FFT magnitude.
    """

    if freqs.size == 0 or amplitude.size == 0:
        return [
            {
                "label": label,
                "low_hz": low,
                "high_hz": high,
                "power": 0.0,
                "fraction": 0.0,
            }
            for low, high, label in bands
        ]

    valid = (freqs >= min_hz) & (freqs <= max_hz)
    power = amplitude * amplitude
    total = float(np.sum(power[valid])) if np.any(valid) else 0.0

    out: list[dict[str, float | str]] = []
    for low, high, label in bands:
        m = (freqs >= low) & (freqs < high) & valid
        p = float(np.sum(power[m])) if np.any(m) else 0.0
        frac = (p / total) if total > 0 else 0.0
        out.append({"label": label, "low_hz": low, "high_hz": high, "power": p, "fraction": frac})
    return out


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
