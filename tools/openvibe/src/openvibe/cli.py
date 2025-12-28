from __future__ import annotations

import argparse
from pathlib import Path

from openvibe import __version__
from openvibe.analysis import (
    load_data,
    maybe_plot,
)
from openvibe.reporting import analyze_df, write_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze elevator ride vibration data.",
        epilog=(
            "Examples:\n"
            "  openvibe ride.csv --units g --plot\n"
            "  openvibe today.csv --baseline baseline.csv --units g\n"
            "  openvibe ride.csv --output-subdir\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"openvibe {__version__}")
    parser.add_argument("csv", type=Path, help="CSV file with timestamp, ax, ay, az columns.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for reports (default: alongside CSV).",
    )
    parser.add_argument(
        "--output-subdir",
        action="store_true",
        help="Write outputs to a dedicated subfolder (<csv_stem>_openvibe) when --output isn't set.",
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
    if args.output is None and args.output_subdir:
        output_dir = args.csv.parent / f"{args.csv.stem}_openvibe"

    sample_rate, duration, peaks, metrics, band_summary = analyze_df(
        df, max_peaks=args.max_peaks, units=args.units
    )

    # Plot uses the same FFT path as analysis (recompute for clarity; still fast for our data sizes).
    plot_path = None
    if args.plot:
        from openvibe.analysis import compute_fft

        freqs, amplitude = compute_fft(df, sample_rate)
        plot_path = maybe_plot(freqs, amplitude, output_dir)
        if plot_path is None:
            print("Note: --plot requested but matplotlib isn't available. Install: openvibe[plot]")

    baseline_csv = None
    baseline_peaks = None
    baseline_metrics = None
    baseline_band_summary = None
    if args.baseline is not None:
        baseline_csv = args.baseline
        baseline_df = load_data(baseline_csv)
        _, _, baseline_peaks, baseline_metrics, baseline_band_summary = analyze_df(
            baseline_df, max_peaks=args.max_peaks, units=args.units
        )

    md_path, json_path, payload = write_reports(
        output_dir=output_dir,
        input_csv=args.csv,
        units=args.units,
        sample_rate_hz=sample_rate,
        duration_seconds=duration,
        peaks=peaks,
        metrics=metrics,
        band_summary=band_summary,
        plot_path=plot_path,
        baseline_csv=baseline_csv,
        baseline_peaks=baseline_peaks,
        baseline_metrics=baseline_metrics,
        baseline_band_summary=baseline_band_summary,
    )

    # Friendly console summary (so you don't have to open report.md every time).
    print(f"Wrote: {md_path} and {json_path}")
    if plot_path is not None:
        print(f"Wrote: {plot_path}")

    top_band = max(band_summary, key=lambda b: float(b["fraction"])) if band_summary else None
    if top_band is not None:
        print(f"Top band: {top_band['label']} (fraction {float(top_band['fraction']):.3f})")

    if peaks:
        p0 = peaks[0]
        print(f"Top peak: {p0.frequency:.2f} Hz ({p0.issue})")
    else:
        print("No significant vibration peaks detected. Consider recording a longer trace.")

    print(
        "Key metrics: "
        f"accel_rms={metrics.accel_rms_vector:.4f} m/s², "
        f"jerk_rms={metrics.jerk_rms_vector:.4f} m/s³"
    )

    baseline = payload.get("baseline")
    if isinstance(baseline, dict) and "delta_spectral_bands" in baseline:
        deltas = baseline["delta_spectral_bands"]
        if isinstance(deltas, list) and deltas:
            worst = max(deltas, key=lambda b: abs(float(b.get("delta_fraction", 0.0))))
            print(
                "Biggest band delta: "
                f"{worst.get('label')} ({float(worst.get('delta_fraction', 0.0)):+.3f})"
            )


if __name__ == "__main__":
    main()
