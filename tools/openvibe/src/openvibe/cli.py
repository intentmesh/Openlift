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
    parser.add_argument(
        "csv",
        nargs="+",
        type=Path,
        help="One or more CSV files with timestamp, ax, ay, az columns.",
    )
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
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Write an aggregated summary CSV (useful for batch runs).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Baseline is computed once and applied to all inputs (if provided).
    baseline_csv = args.baseline
    baseline_peaks = None
    baseline_metrics = None
    baseline_band_summary = None
    baseline_load_stats = None
    if baseline_csv is not None:
        baseline_df = load_data(baseline_csv)
        (
            _,
            _,
            baseline_peaks,
            baseline_metrics,
            baseline_band_summary,
            baseline_load_stats,
        ) = analyze_df(baseline_df, max_peaks=args.max_peaks, units=args.units)

    summaries: list[dict[str, object]] = []

    for csv_path in args.csv:
        df = load_data(csv_path)

        # Output layout:
        # - If --output is set: always write per-input subfolders inside it.
        # - Else if multiple inputs: write per-input subfolders next to each CSV.
        # - Else: maintain legacy behavior unless --output-subdir is set.
        if args.output is not None:
            output_dir = args.output / f"{csv_path.stem}_openvibe"
        elif len(args.csv) > 1 or args.output_subdir:
            output_dir = csv_path.parent / f"{csv_path.stem}_openvibe"
        else:
            output_dir = csv_path.parent

        sample_rate, duration, peaks, metrics, band_summary, load_stats = analyze_df(
            df, max_peaks=args.max_peaks, units=args.units
        )

        # Plot uses the same FFT path as analysis (recompute for clarity; still fast for our data sizes).
        plot_path = None
        if args.plot:
            from openvibe.analysis import compute_fft

            freqs, amplitude = compute_fft(df, sample_rate)
            plot_path = maybe_plot(freqs, amplitude, output_dir)
            if plot_path is None:
                print(
                    "Note: --plot requested but matplotlib isn't available. Install: openvibe[plot]"
                )

        md_path, json_path, payload = write_reports(
            output_dir=output_dir,
            input_csv=csv_path,
            units=args.units,
            sample_rate_hz=sample_rate,
            duration_seconds=duration,
            peaks=peaks,
            metrics=metrics,
            band_summary=band_summary,
            load_stats=load_stats,
            plot_path=plot_path,
            baseline_csv=baseline_csv,
            baseline_peaks=baseline_peaks,
            baseline_metrics=baseline_metrics,
            baseline_band_summary=baseline_band_summary,
            baseline_load_stats=baseline_load_stats,
        )

        # Friendly per-file console summary.
        print(f"\n== {csv_path.name} ==")
        print(f"Wrote: {md_path} and {json_path}")
        if plot_path is not None:
            print(f"Wrote: {plot_path}")
        if load_stats:
            print(
                "Data quality: "
                f"raw_rows={int(load_stats.get('raw_rows', 0))}, "
                f"dropped={int(load_stats.get('rows_dropped_non_numeric_or_na', 0))}, "
                f"dupe_ts_dropped={int(load_stats.get('duplicate_timestamps_dropped', 0))}, "
                f"final_rows={int(load_stats.get('final_rows', 0))}"
            )

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

        summaries.append(
            {
                "input_csv": str(csv_path),
                "output_dir": str(output_dir),
                "sample_rate_hz": sample_rate,
                "duration_seconds": duration,
                "final_rows": int(load_stats.get("final_rows", 0)),
                "accel_rms_vector_mps2": metrics.accel_rms_vector,
                "jerk_rms_vector_mps3": metrics.jerk_rms_vector,
                "top_band": str(top_band["label"]) if top_band else "",
                "top_band_fraction": float(top_band["fraction"]) if top_band else 0.0,
                "top_peak_hz": peaks[0].frequency if peaks else 0.0,
                "top_peak_issue": peaks[0].issue if peaks else "",
            }
        )

    if args.summary_csv is not None:
        import csv as _csv

        out = args.summary_csv
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for row in summaries for k in row.keys()})
        with out.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summaries)
        print(f"\nWrote summary CSV: {out}")


if __name__ == "__main__":
    main()
