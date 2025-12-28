from __future__ import annotations

import argparse
from pathlib import Path

from openvibe.analysis import (
    load_data,
    maybe_plot,
)
from openvibe.reporting import analyze_df, write_reports


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
    parser.add_argument(
        "--units",
        default="m/s2",
        choices=["m/s2", "m/s^2", "mps2", "ms2", "g"],
        help="Units of ax/ay/az in the CSV (default: m/s2).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline CSV to compare against (same format/units).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    df = load_data(args.csv)
    output_dir = args.output or args.csv.parent

    sample_rate, duration, peaks, metrics = analyze_df(
        df, max_peaks=args.max_peaks, units=args.units
    )

    # Plot uses the same FFT path as analysis (recompute for clarity; still fast for our data sizes).
    plot_path = None
    if args.plot:
        from openvibe.analysis import compute_fft

        freqs, amplitude = compute_fft(df, sample_rate)
        plot_path = maybe_plot(freqs, amplitude, output_dir)

    baseline_csv = None
    baseline_peaks = None
    baseline_metrics = None
    if args.baseline is not None:
        baseline_csv = args.baseline
        baseline_df = load_data(baseline_csv)
        _, _, baseline_peaks, baseline_metrics = analyze_df(
            baseline_df, max_peaks=args.max_peaks, units=args.units
        )

    write_reports(
        output_dir=output_dir,
        input_csv=args.csv,
        units=args.units,
        sample_rate_hz=sample_rate,
        duration_seconds=duration,
        peaks=peaks,
        metrics=metrics,
        plot_path=plot_path,
        baseline_csv=baseline_csv,
        baseline_peaks=baseline_peaks,
        baseline_metrics=baseline_metrics,
    )

    print(f"Report written to {output_dir.resolve()}")
    if not peaks:
        print("No significant vibration peaks detected. Consider recording a longer trace.")


if __name__ == "__main__":
    main()
