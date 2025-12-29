# OpenVibe – Ride Quality Analyzer

OpenVibe is a command-line tool that turns raw accelerometer logs into actionable ride-quality diagnostics aligned with ISO 18738 ride-comfort metrics.

## Features
- Imports CSV sensor data (timestamp + X/Y/Z acceleration in m/s² or g)
- Removes DC offset and applies windowed FFT
- Detects dominant vibration peaks and maps them to common elevator issues
- Outputs a Markdown + JSON report plus optional PNG spectrum plot

## Quick Start

```bash
cd tools/openvibe
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m openvibe.core sample_data.csv --plot
```

### Make/Docker workflows

```bash
# create .venv + install dependencies
make setup

# run analyzer with plotting + JSON stdout
make run

# launch FastAPI service for mobile clients
make serve

# or use Docker
docker build -t openvibe .
docker run --rm -v $PWD:/data openvibe /data/sample_door_resonance.csv --stdout-report
```

## CSV Format

| Column      | Description                          |
|-------------|--------------------------------------|
| `timestamp` | Unix timestamp in seconds (float)    |
| `ax`        | X-axis acceleration (m/s² or g)      |
| `ay`        | Y-axis acceleration                  |
| `az`        | Z-axis acceleration                  |

Use any mobile logging app (e.g., PhyPhox, SensorLog) and export to CSV.

### Logging tips
- Maintain ≥5 s of continuous data per direction of travel.
- Sample at ≥30 Hz (preferably 50–100 Hz).
- Capture all three axes; align phone consistently (X lateral, Y side, Z vertical).
- Note load, speed, and shaft conditions for the report header.

### CLI highlights
- `--units`: `mps2` (default) or `g`
- `--highpass/--lowpass`: adjust filtering
- `--issues-config`: point to custom JSON frequency bands
- `--stdout-report`: emit JSON to stdout for pipelines
- `--segment-*`: configure ride segmentation
- `--cloud-dir`: mirror JSON payloads to a directory (cloud export stub)
- `--plot`: save `spectrum.png`

## Output
- `report.md` – human-readable summary with peak table and remediation hints
- `report.json` – machine-readable metrics
- `spectrum.png` (optional) – FFT magnitude plot

## Issue Database

Frequency bands and recommendations live in [`issues.json`](issues.json). Each entry declares:

```json
{
  "f_low": 9.0,
  "f_high": 14.0,
  "issue": "Guide roller wear",
  "recommendation": "Inspect guide shoes, roller bearings, and lubrication schedule."
}
```

Clone the file, tune ranges for your elevator make/model, and point the CLI to it:

```bash
python openvibe.py ride.csv --issues-config my_issues.json
```

## Testing

```bash
cd tools/openvibe
make setup
.venv/bin/pytest
```

## REST Service
Run the API for remote/mobile ingestion:

```bash
cd tools/openvibe
make serve  # serves http://127.0.0.1:8000
```

Endpoints:
- `GET /health`
- `POST /analyze` (multipart CSV upload) → returns JSON payload identical to CLI output.

## Flutter Companion
A Flutter skeleton lives in [`mobile/openvibe_mobile`](mobile/openvibe_mobile). It streams accelerometer data, shows instantaneous RMS, and hits the REST endpoint for full analysis. This will evolve into the production mobile app.

## Next Steps
- Live mobile app (Flutter) streaming directly from device sensors
- ISO 18738 ride quality scoring (RMS jerk, comfort index)
- Automated PDF export for modernization proposals



