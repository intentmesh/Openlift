from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile

from openvibe import AnalyzerOptions, DEFAULT_ISSUES_PATH, analyze_file

app = FastAPI(title="OpenVibe Service", version="1.0.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    units: str = "mps2",
    highpass: float = 0.3,
    lowpass: float = 40.0,
    max_peaks: int = 5,
    segment_threshold: float = 0.2,
    segment_min_duration: float = 2.0,
    enable_segments: bool = True,
) -> dict:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / file.filename
        content = await file.read()
        tmp_path.write_bytes(content)

        options = AnalyzerOptions(
            max_peaks=max_peaks,
            units=units,
            highpass=highpass,
            lowpass=lowpass,
            segment_threshold=segment_threshold,
            segment_min_duration=segment_min_duration,
            enable_segments=enable_segments,
            issues_path=DEFAULT_ISSUES_PATH,
        )
        payload = analyze_file(
            tmp_path,
            options,
            output_dir=Path(tmpdir),
            include_plot=False,
            persist_reports=False,
        )
        return payload

