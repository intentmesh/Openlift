# OpenVibe – Ride Quality Analyzer

OpenVibe is a command-line tool that turns raw accelerometer logs into actionable ride-quality diagnostics aligned with ISO 18738 ride-comfort metrics.

## Features
- Imports CSV sensor data (timestamp + X/Y/Z acceleration in m/s² or g)
- Removes DC offset and applies windowed FFT
- Detects dominant vibration peaks and maps them to common elevator issues
- Computes time-domain ride metrics (acceleration + jerk)
- Handles messy real-world exports (column aliases, delimiter sniffing, timestamp unit auto-detect, gap/jitter stats)
- Outputs a Markdown + JSON report plus optional PNG spectrum plot

## Quick Start

```bash
cd tools/openvibe
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[plot]"
openvibe sample_data.csv --plot --output-subdir
```

## Batch Mode (multiple files)

Analyze multiple CSVs and write one summary table:

```bash
openvibe ride1.csv ride2.csv ride3.csv --units g --summary-csv summary.csv
```

Batch outputs are written to per-run folders using a stable `run_id` so files won't overwrite each other.

## Robust CSV Ingestion

- **Column aliases**: headers like `Time`, `Accel X (m/s^2)` etc are accepted.
- **Delimiters**: comma / semicolon / tab are auto-detected.
- **Timestamp units**: auto-detected; override if needed:

```bash
openvibe ride.csv --timestamp-unit ms
```

## Fingerprints + Baselines + Scoring (new)

Create a fingerprint (compact feature vector) for later comparison:

```bash
openvibe fingerprint ride.csv --units g --out .
```

Add fingerprints to a baseline group and build a model:

```bash
openvibe baseline add *.fingerprint.json --tag elevator=E12
openvibe baseline build --tag elevator=E12
```

Score a new ride against that baseline (explainable anomaly score):

```bash
openvibe score today.csv --tag elevator=E12
```

For field reliability, scoring is most meaningful once you have a baseline of ~5+ rides:

```bash
openvibe score today.csv --tag elevator=E12 --min-samples 5
```

### Legacy (no install)

```bash
python openvibe.py sample_data.csv --plot
```

## CSV Format

| Column      | Description                          |
|-------------|--------------------------------------|
| `timestamp` | Unix timestamp in seconds (float)    |
| `ax`        | X-axis acceleration (m/s² or g)      |
| `ay`        | Y-axis acceleration                  |
| `az`        | Z-axis acceleration                  |

Use any mobile logging app (e.g., PhyPhox, SensorLog) and export to CSV.

## Output
- `report.md` – human-readable summary with peak table and remediation hints
- `report.json` – machine-readable metrics
- `spectrum.png` (optional) – FFT magnitude plot

## Baseline Comparison

Use a known-good ride trace as a baseline to get deltas:

```bash
openvibe today.csv --baseline baseline.csv --units g
```

## Tips
- Use `--output-subdir` to avoid overwriting previous runs.
- Run `openvibe --help` to see all options (including `--version`).

## Dev Housekeeping

If you're making changes, enable pre-commit so lint/format stay clean:

```bash
pip install -e ".[dev]"
pre-commit install
```

## Next Steps
- Live mobile app (Flutter) streaming directly from device sensors
- ISO 18738 ride quality scoring (RMS jerk, comfort index)
- Automated PDF export for modernization proposals

