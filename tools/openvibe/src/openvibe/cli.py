from __future__ import annotations

import argparse
import csv as _csv
import json
import sys
from pathlib import Path

from openvibe import __version__
from openvibe.analysis import load_data, maybe_plot
from openvibe.baselines import (
    add_fingerprint,
    build_model,
    default_store_path,
    get_or_build_model,
    group_key,
    list_groups,
    score_features,
)
from openvibe.fingerprint import (
    build_fingerprint,
    read_fingerprint,
    stable_run_id,
    write_fingerprint,
)
from openvibe.reporting import analyze_df, write_reports


def _parse_tags(pairs: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"Invalid tag {p!r}; expected key=value")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


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

    sub = parser.add_subparsers(dest="command")

    # analyze (default command for backwards compatibility)
    analyze = sub.add_parser("analyze", help="Generate report.md/report.json from CSVs.")
    analyze.add_argument("csv", nargs="+", type=Path, help="One or more CSV files.")
    analyze.add_argument("--output", type=Path, default=None, help="Output directory (optional).")
    analyze.add_argument(
        "--output-subdir",
        action="store_true",
        help="Write outputs to a dedicated subfolder when --output isn't set.",
    )
    analyze.add_argument("--plot", action="store_true", help="Generate spectrum plot.")
    analyze.add_argument("--max-peaks", type=int, default=4, help="Number of peaks to report.")
    analyze.add_argument(
        "--units",
        default="m/s2",
        choices=["m/s2", "m/s^2", "mps2", "ms2", "g"],
        help="Units of ax/ay/az in the CSV (default: m/s2).",
    )
    analyze.add_argument(
        "--timestamp-unit",
        default="auto",
        choices=["auto", "s", "ms", "us", "ns"],
        help="Timestamp unit in the CSV (default: auto-detect).",
    )
    analyze.add_argument("--baseline", type=Path, default=None, help="Baseline CSV (optional).")
    analyze.add_argument(
        "--summary-csv", type=Path, default=None, help="Write summary CSV (batch)."
    )

    # fingerprint
    fp = sub.add_parser("fingerprint", help="Write a compact fingerprint.json for each CSV.")
    fp.add_argument("csv", nargs="+", type=Path, help="One or more CSV files.")
    fp.add_argument("--out", type=Path, default=None, help="Output directory (optional).")
    fp.add_argument("--max-peaks", type=int, default=4)
    fp.add_argument(
        "--units",
        default="m/s2",
        choices=["m/s2", "m/s^2", "mps2", "ms2", "g"],
    )
    fp.add_argument(
        "--timestamp-unit",
        default="auto",
        choices=["auto", "s", "ms", "us", "ns"],
    )

    # baseline
    baseline = sub.add_parser("baseline", help="Manage local baseline store.")
    baseline_sub = baseline.add_subparsers(dest="baseline_cmd", required=True)

    baseline_add = baseline_sub.add_parser("add", help="Add fingerprint(s) to a group.")
    baseline_add.add_argument("fingerprint", nargs="+", type=Path)
    baseline_add.add_argument("--store", type=Path, default=default_store_path())
    baseline_add.add_argument(
        "--tag", action="append", default=[], help="Group tag key=value", dest="tags"
    )

    baseline_build = baseline_sub.add_parser("build", help="Build/refresh model for a group.")
    baseline_build.add_argument("--store", type=Path, default=default_store_path())
    baseline_build.add_argument(
        "--tag", action="append", default=[], help="Group tag key=value", dest="tags"
    )

    baseline_list = baseline_sub.add_parser("list", help="List groups.")
    baseline_list.add_argument("--store", type=Path, default=default_store_path())

    # score
    score = sub.add_parser("score", help="Score CSVs against a baseline group.")
    score.add_argument("csv", nargs="+", type=Path)
    score.add_argument("--store", type=Path, default=default_store_path())
    score.add_argument(
        "--tag", action="append", default=[], help="Group tag key=value", dest="tags"
    )
    score.add_argument("--json", action="store_true", help="Print JSON output.")
    score.add_argument("--max-peaks", type=int, default=4)
    score.add_argument(
        "--units",
        default="m/s2",
        choices=["m/s2", "m/s^2", "mps2", "ms2", "g"],
    )
    score.add_argument(
        "--timestamp-unit",
        default="auto",
        choices=["auto", "s", "ms", "us", "ns"],
    )

    return parser


def _cmd_analyze(args: argparse.Namespace) -> None:
    baseline_csv = args.baseline
    baseline_peaks = None
    baseline_metrics = None
    baseline_band_summary = None
    baseline_load_stats = None
    if baseline_csv is not None:
        baseline_df = load_data(baseline_csv, timestamp_unit=args.timestamp_unit)
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
        df = load_data(csv_path, timestamp_unit=args.timestamp_unit)
        run_id = stable_run_id(csv_path)

        if args.output is not None:
            output_dir = args.output / f"{run_id}_openvibe"
        elif len(args.csv) > 1 or args.output_subdir:
            output_dir = csv_path.parent / f"{run_id}_openvibe"
        else:
            output_dir = csv_path.parent

        sample_rate, duration, peaks, metrics, band_summary, load_stats = analyze_df(
            df, max_peaks=args.max_peaks, units=args.units
        )

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
            run_id=run_id,
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
                f"final_rows={int(load_stats.get('final_rows', 0))}, "
                f"dt_median_s={float(load_stats.get('dt_median_s', 0.0)):.6f}, "
                f"gap_count={int(load_stats.get('gap_count', 0))}"
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
                "run_id": run_id,
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
        out = args.summary_csv
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for row in summaries for k in row.keys()})
        with out.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summaries)
        print(f"\nWrote summary CSV: {out}")


def _cmd_fingerprint(args: argparse.Namespace) -> None:
    for csv_path in args.csv:
        df = load_data(csv_path, timestamp_unit=args.timestamp_unit)
        fp = build_fingerprint(df, input_csv=csv_path, units=args.units, max_peaks=args.max_peaks)

        out_dir = args.out or csv_path.parent
        out_path = out_dir / f"{fp.run_id}.fingerprint.json"
        write_fingerprint(fp, out_path)
        print(f"Wrote fingerprint: {out_path}")


def _cmd_baseline(args: argparse.Namespace) -> None:
    store_path: Path = args.store
    if args.baseline_cmd == "list":
        for g in list_groups(store_path):
            print(f"{g['group']}: {g['samples']} samples")
        return

    tags = _parse_tags(getattr(args, "tags", []))
    if args.baseline_cmd == "add":
        for fp_path in args.fingerprint:
            fp = read_fingerprint(fp_path)
            add_fingerprint(store_path=store_path, tags=tags, fingerprint=fp)
            print(f"Added: {fp_path} -> group {group_key(tags)}")
        return

    if args.baseline_cmd == "build":
        model = build_model(store_path=store_path, tags=tags)
        print(f"Built baseline model for {group_key(tags)} (n={model.n})")
        return

    raise ValueError(f"Unknown baseline command: {args.baseline_cmd}")


def _cmd_score(args: argparse.Namespace) -> None:
    tags = _parse_tags(args.tags)
    model = get_or_build_model(store_path=args.store, tags=tags)
    out_all: list[dict[str, object]] = []

    for csv_path in args.csv:
        df = load_data(csv_path, timestamp_unit=args.timestamp_unit)
        fp = build_fingerprint(df, input_csv=csv_path, units=args.units, max_peaks=args.max_peaks)
        score, contributors = score_features(features=fp.features, model=model)
        row = {
            "run_id": fp.run_id,
            "input_csv": fp.input_csv,
            "group": group_key(tags),
            "baseline_n": model.n,
            "anomaly_score": score,
            "top_contributors": contributors,
        }
        out_all.append(row)

        if not args.json:
            print(f"\n== {csv_path.name} ==")
            print(f"Group: {group_key(tags)} (baseline n={model.n})")
            print(f"Anomaly score: {score:.1f}/100")
            if contributors:
                top = ", ".join(
                    f"{c['feature']} ({c['contribution']:.2f})" for c in contributors[:5]
                )
                print(f"Top contributors: {top}")

    if args.json:
        print(json.dumps(out_all, indent=2))


def main(argv: list[str] | None = None) -> None:
    argv = list(argv) if argv is not None else sys.argv[1:]
    # Backwards compatibility: `openvibe file.csv ...` == `openvibe analyze file.csv ...`
    if argv and argv[0] not in {
        "analyze",
        "fingerprint",
        "baseline",
        "score",
        "--version",
        "-h",
        "--help",
    }:
        argv = ["analyze", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "fingerprint":
        _cmd_fingerprint(args)
    elif args.command == "baseline":
        _cmd_baseline(args)
    elif args.command == "score":
        _cmd_score(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
