from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

G0_MPS2 = 9.80665


class UnitsError(ValueError):
    pass


def units_scale_to_mps2(units: str) -> float:
    """
    Return scale factor that converts the provided units to m/s^2.

    - "m/s2", "m/s^2", "mps2" => 1.0
    - "g" => 9.80665
    """

    u = units.strip().lower()
    if u in {"m/s2", "m/s^2", "mps2", "ms2"}:
        return 1.0
    if u in {"g"}:
        return G0_MPS2
    raise UnitsError(f"Unsupported units: {units!r}. Expected 'm/s2' or 'g'.")


@dataclass(frozen=True)
class TimeMetrics:
    units: str
    dc_removed: bool
    accel_rms_x: float
    accel_rms_y: float
    accel_rms_z: float
    accel_rms_vector: float
    jerk_rms_x: float
    jerk_rms_y: float
    jerk_rms_z: float
    jerk_rms_vector: float
    accel_p95_vector: float
    jerk_p95_vector: float

    def to_json(self) -> dict[str, float | str]:
        return {
            "units": self.units,
            "dc_removed": self.dc_removed,
            "accel_rms_x": self.accel_rms_x,
            "accel_rms_y": self.accel_rms_y,
            "accel_rms_z": self.accel_rms_z,
            "accel_rms_vector": self.accel_rms_vector,
            "jerk_rms_x": self.jerk_rms_x,
            "jerk_rms_y": self.jerk_rms_y,
            "jerk_rms_z": self.jerk_rms_z,
            "jerk_rms_vector": self.jerk_rms_vector,
            "accel_p95_vector": self.accel_p95_vector,
            "jerk_p95_vector": self.jerk_p95_vector,
        }


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(x * x))))


def _pctl(x: np.ndarray, p: float) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, p))


def compute_time_metrics(df: pd.DataFrame, *, units: str, remove_dc: bool = True) -> TimeMetrics:
    """
    Compute time-domain ride metrics from raw accelerometer axes.

    Notes:
    - Acceleration is converted to m/s^2 (or treated as already m/s^2) based on `units`.
    - By default, DC is removed per-axis (helps avoid gravity/tilt dominating vibration metrics).
    - Jerk is computed via numerical gradient against the provided timestamps.
    - Vector metrics are computed as sqrt(x^2+y^2+z^2) of the corresponding quantity.
    """

    scale = units_scale_to_mps2(units)
    t = df["timestamp"].to_numpy(dtype=float)
    ax = df["ax"].to_numpy(dtype=float) * scale
    ay = df["ay"].to_numpy(dtype=float) * scale
    az = df["az"].to_numpy(dtype=float) * scale

    if remove_dc:
        ax = ax - float(np.mean(ax)) if ax.size else ax
        ay = ay - float(np.mean(ay)) if ay.size else ay
        az = az - float(np.mean(az)) if az.size else az

    a_vec = np.sqrt(ax * ax + ay * ay + az * az)

    if t.size >= 2:
        jx = np.gradient(ax, t)
        jy = np.gradient(ay, t)
        jz = np.gradient(az, t)
    else:
        jx = np.zeros_like(ax)
        jy = np.zeros_like(ay)
        jz = np.zeros_like(az)

    j_vec = np.sqrt(jx * jx + jy * jy + jz * jz)

    return TimeMetrics(
        units="m/s2",
        dc_removed=remove_dc,
        accel_rms_x=_rms(ax),
        accel_rms_y=_rms(ay),
        accel_rms_z=_rms(az),
        accel_rms_vector=_rms(a_vec),
        jerk_rms_x=_rms(jx),
        jerk_rms_y=_rms(jy),
        jerk_rms_z=_rms(jz),
        jerk_rms_vector=_rms(j_vec),
        accel_p95_vector=_pctl(a_vec, 95.0),
        jerk_p95_vector=_pctl(j_vec, 95.0),
    )
