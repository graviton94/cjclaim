#!/usr/bin/env python3
"""
Normalize trained models snapshot and generate review CSVs.

Writes:
 - artifacts/trained_models_part.json.bak.TIMESTAMP (backup)
 - artifacts/trained_models_part.json (overwritten with normalized content)
 - artifacts/trained_models_part_normalized.json (copy)
 - artifacts/selection_review.csv (summary per series)
 - artifacts/seasonal_trained_list.csv (seasonal series only)
 - artifacts/missing_aic_list.csv (series missing aic)

Run from repository root.
"""
from __future__ import annotations

import json
import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SNAP = ARTIFACTS / "trained_models_part.json"


def load_snapshot(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    sm = data.get("series_models", {})
    for name, entry in sm.items():
        # ensure keys exist
        if "aic" not in entry:
            entry["aic"] = None
        if "arroots" not in entry:
            entry["arroots"] = None
    data["series_models"] = sm
    return data


def is_seasonal(entry: dict[str, Any]) -> bool:
    so = entry.get("model_spec", {}).get("seasonal_order")
    if not so:
        return False
    return any(int(x) != 0 for x in so)


def make_csvs(data: dict[str, Any]) -> None:
    sm = data.get("series_models", {})
    review_path = ARTIFACTS / "selection_review.csv"
    seasonal_path = ARTIFACTS / "seasonal_trained_list.csv"
    missing_aic_path = ARTIFACTS / "missing_aic_list.csv"

    with review_path.open("w", encoding="utf-8-sig", newline="") as rf:
        writer = csv.writer(rf)
        writer.writerow([
            "series",
            "n_train_points",
            "order",
            "seasonal_order",
            "is_seasonal",
            "aic",
            "selection_loss",
            "hist_mean",
            "hist_std",
            "last_value",
        ])
        for name, e in sm.items():
            mo = e.get("model_spec", {}).get("order")
            so = e.get("model_spec", {}).get("seasonal_order")
            writer.writerow([
                name,
                e.get("n_train_points"),
                json.dumps(mo, ensure_ascii=False),
                json.dumps(so, ensure_ascii=False),
                is_seasonal(e),
                e.get("aic"),
                e.get("selection_loss"),
                e.get("hist_mean"),
                e.get("hist_std"),
                e.get("last_value"),
            ])

    with seasonal_path.open("w", encoding="utf-8-sig", newline="") as sf:
        writer = csv.writer(sf)
        writer.writerow(["series", "aic", "n_train_points", "seasonal_order"])
        for name, e in sm.items():
            if is_seasonal(e):
                writer.writerow([name, e.get("aic"), e.get("n_train_points"), json.dumps(e.get("model_spec", {}).get("seasonal_order"), ensure_ascii=False)])

    with missing_aic_path.open("w", encoding="utf-8-sig", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["series", "n_train_points", "model_spec"])
        for name, e in sm.items():
            if e.get("aic") is None:
                writer.writerow([name, e.get("n_train_points"), json.dumps(e.get("model_spec", {}), ensure_ascii=False)])


def main() -> None:
    if not SNAP.exists():
        print(f"Snapshot not found: {SNAP}")
        return
    data = load_snapshot(SNAP)
    # backup
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bak = ARTIFACTS / f"trained_models_part.json.bak.{ts}"
    shutil.copy2(SNAP, bak)
    print(f"Backup written: {bak}")

    normalized = normalize_snapshot(data)
    # overwrite original and also write normalized copy
    write_json(SNAP, normalized)
    copy_path = ARTIFACTS / "trained_models_part_normalized.json"
    write_json(copy_path, normalized)
    print(f"Wrote normalized snapshot: {SNAP} and copy: {copy_path}")

    make_csvs(normalized)
    print("Wrote CSVs: selection_review.csv, seasonal_trained_list.csv, missing_aic_list.csv")

    # summary counts
    sm = normalized.get("series_models", {})
    total = len(sm)
    missing_aic = sum(1 for e in sm.values() if e.get("aic") is None)
    seasonal = sum(1 for e in sm.values() if is_seasonal(e))
    print(f"Total series: {total}")
    print(f"Missing aic: {missing_aic}")
    print(f"Seasonal models: {seasonal}")


if __name__ == "__main__":
    main()
