from __future__ import annotations

import argparse
from pathlib import Path

from openvibe.analysis import (
    compute_fft,
    estimate_sample_rate,
    find_peaks,
    load_data,
    maybe_plot,
    write_reports,
)


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    df = load_data(args.csv)
    output_dir = args.output or args.csv.parent

    sample_rate = estimate_sample_rate(df)
    freqs, amplitude = compute_fft(df, sample_rate)
    peaks = find_peaks(freqs, amplitude, args.max_peaks)
    duration = float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) if len(df) else 0.0

    plot_path = maybe_plot(freqs, amplitude, output_dir) if args.plot else None
    write_reports(output_dir, peaks, sample_rate, duration, plot_path)

    print(f"Report written to {output_dir.resolve()}")
    if not peaks:
        print("No significant vibration peaks detected. Consider recording a longer trace.")


if __name__ == "__main__":
    main()

