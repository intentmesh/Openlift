# OpenVibe – Ride Quality Analyzer

OpenVibe is a command-line tool that turns raw accelerometer logs into actionable ride-quality diagnostics aligned with ISO 18738 ride-comfort metrics.

## Features
- Imports CSV sensor data (timestamp + X/Y/Z acceleration in m/s² or g)
- Removes DC offset and applies windowed FFT
- Detects dominant vibration peaks and maps them to common elevator issues
- Computes time-domain ride metrics (acceleration + jerk)
- Outputs a Markdown + JSON report plus optional PNG spectrum plot

## Quick Start

```bash
cd tools/openvibe
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[plot]"
openvibe sample_data.csv --plot
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

