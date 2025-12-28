from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

BASELINES_SCHEMA_VERSION = "openvibe.baselines.v1"


def default_store_path() -> Path:
    return Path(".openvibe") / "baselines.json"


def group_key(tags: dict[str, str]) -> str:
    items = sorted(tags.items())
    return ",".join(f"{k}={v}" for k, v in items) if items else "default"


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _mad(xs: list[float], med: float) -> float:
    return _median([abs(x - med) for x in xs])


@dataclass(frozen=True)
class BaselineModel:
    center: dict[str, float]
    mad: dict[str, float]
    n: int

    def to_json(self) -> dict[str, object]:
        return {"center": self.center, "mad": self.mad, "n": self.n}


def load_store(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"schema_version": BASELINES_SCHEMA_VERSION, "groups": {}}
    store = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(store, dict):
        raise ValueError("Invalid baselines store: root must be an object")
    sv = store.get("schema_version")
    if sv != BASELINES_SCHEMA_VERSION:
        raise ValueError(f"Unsupported baselines store schema_version: {sv!r}")
    groups = store.get("groups")
    if groups is None:
        store["groups"] = {}
    elif not isinstance(groups, dict):
        raise ValueError("Invalid baselines store: groups must be an object")
    return store


def save_store(path: Path, store: dict[str, object]) -> None:
    """
    Field-grade: atomic write to prevent corruption on power loss.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(store, indent=2)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", dir=str(path.parent), prefix=path.name, suffix=".tmp"
    ) as f:
        tmp = Path(f.name)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def add_fingerprint(
    *,
    store_path: Path,
    tags: dict[str, str],
    fingerprint: dict[str, object],
) -> None:
    store = load_store(store_path)
    groups = store.setdefault("groups", {})

    gk = group_key(tags)
    group_obj = groups.setdefault(gk, {"samples": [], "model": None})
    samples = group_obj.setdefault("samples", [])
    if not isinstance(samples, list):
        raise ValueError("Invalid baselines store: samples must be a list")

    features = fingerprint.get("features")
    if not isinstance(features, dict):
        raise ValueError("Invalid fingerprint: missing features object")

    samples.append(
        {
            "added_at_epoch": time.time(),
            "run_id": fingerprint.get("run_id"),
            "input": fingerprint.get("input"),
            "tool_version": fingerprint.get("tool_version"),
            "features": features,
        }
    )
    # Model becomes stale when adding new samples.
    group_obj["model"] = None
    save_store(store_path, store)


def build_model(*, store_path: Path, tags: dict[str, str]) -> BaselineModel:
    store = load_store(store_path)
    groups = store.get("groups", {})

    gk = group_key(tags)
    group_obj = groups.get(gk)
    if not isinstance(group_obj, dict):
        raise ValueError(f"No baseline group found: {gk}")
    samples = group_obj.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"No samples for baseline group: {gk}")

    # Collect per-feature values across samples.
    values: dict[str, list[float]] = {}
    for s in samples:
        if not isinstance(s, dict):
            continue
        feats = s.get("features")
        if not isinstance(feats, dict):
            continue
        for k, v in feats.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                values.setdefault(k, []).append(float(v))

    center: dict[str, float] = {}
    mad: dict[str, float] = {}
    for k, xs in values.items():
        med = _median(xs)
        center[k] = med
        mad[k] = _mad(xs, med)

    model = BaselineModel(center=center, mad=mad, n=len(samples))
    group_obj["model"] = model.to_json()
    save_store(store_path, store)
    return model


def get_or_build_model(*, store_path: Path, tags: dict[str, str]) -> BaselineModel:
    store = load_store(store_path)
    groups = store.get("groups", {})

    gk = group_key(tags)
    group_obj = groups.get(gk)
    if not isinstance(group_obj, dict):
        raise ValueError(f"No baseline group found: {gk}")

    model = group_obj.get("model")
    if isinstance(model, dict):
        center = model.get("center")
        mad = model.get("mad")
        n = model.get("n")
        if isinstance(center, dict) and isinstance(mad, dict) and isinstance(n, int):
            return BaselineModel(
                center={str(k): float(v) for k, v in center.items()},
                mad={str(k): float(v) for k, v in mad.items()},
                n=n,
            )
    return build_model(store_path=store_path, tags=tags)


def list_groups(store_path: Path) -> list[dict[str, object]]:
    store = load_store(store_path)
    groups = store.get("groups", {})
    if not isinstance(groups, dict):
        return []
    out: list[dict[str, object]] = []
    for gk, obj in groups.items():
        if not isinstance(obj, dict):
            continue
        samples = obj.get("samples", [])
        out.append({"group": gk, "samples": len(samples) if isinstance(samples, list) else 0})
    return sorted(out, key=lambda x: str(x["group"]))


def score_features(
    *,
    features: dict[str, float],
    model: BaselineModel,
    weights: dict[str, float] | None = None,
) -> tuple[float, list[dict[str, object]]]:
    """
    Return (score_0_100, top_contributors).

    We use robust z-scores with MAD; overall score is mapped via an exponential curve.
    """

    weights = weights or {}
    eps = 1e-9
    contributions: list[tuple[str, float]] = []
    z2_sum = 0.0
    n = 0
    for k, x in features.items():
        if k not in model.center:
            continue
        w = float(weights.get(k, 1.0))
        if w <= 0:
            continue
        med = model.center[k]
        mad = model.mad.get(k, 0.0)
        # Convert MAD to sigma-ish scale (Normal approx).
        sigma = max(eps, 1.4826 * mad)
        z = (x - med) / sigma
        z2_sum += (w * z) ** 2
        n += 1
        contributions.append((k, abs(w * z)))

    rms_z = math.sqrt(z2_sum / max(1, n))
    score = 100.0 * (1.0 - math.exp(-rms_z / 2.0))
    score = max(0.0, min(100.0, score))

    top = sorted(contributions, key=lambda kv: kv[1], reverse=True)[:8]
    top_out = [{"feature": k, "contribution": c} for k, c in top]
    return score, top_out
