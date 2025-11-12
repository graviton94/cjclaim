import json
from pathlib import Path
import csv
import datetime


RERUN_SNAPSHOT = Path("artifacts/trained_models_v3_rerun_500.json")
ORIG_SNAPSHOT = Path("artifacts/trained_models_v3.json")
VALIDATION_CSV = Path("artifacts/validation_seasonal_flag_changes.csv")
OUT_SNAPSHOT = Path("artifacts/trained_models_v3_rerun_500_arroot_filtered.json")
SUMMARY_CSV = Path("artifacts/arroot_filter_summary.csv")

THRESHOLD = 1.0


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if not RERUN_SNAPSHOT.exists():
        raise SystemExit(f"Missing rerun snapshot: {RERUN_SNAPSHOT}")

    rerun = load_json(RERUN_SNAPSHOT)
    orig = load_json(ORIG_SNAPSHOT) if ORIG_SNAPSHOT.exists() else None

    # read validation CSV into dict keyed by series_id
    flags = {}
    if VALIDATION_CSV.exists():
        with VALIDATION_CSV.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                sid = r.get("series_id")
                if not sid:
                    continue
                # parse max_abs_arroot
                try:
                    ar = float(r.get("max_abs_arroot")) if r.get("max_abs_arroot") not in (None, "", "nan") else None
                except Exception:
                    ar = None
                flags[sid] = {
                    "n_train": r.get("n_train"),
                    "orig_seasonal": r.get("orig_seasonal"),
                    "rerun_seasonal": r.get("rerun_seasonal"),
                    "orig_aic": r.get("orig_aic"),
                    "rerun_aic": r.get("rerun_aic"),
                    "aic_delta": r.get("aic_delta"),
                    "max_abs_arroot": ar,
                    "plot": r.get("plot"),
                }

    series_models = rerun.get("series_models", {})
    out_models = dict(series_models)  # shallow copy; we'll replace entries when needed

    summary_rows = []

    for sid, info in flags.items():
        ar = info.get("max_abs_arroot")
        rerun_seasonal = info.get("rerun_seasonal")
        # detect if rerun has seasonal (naive string check for nonzero seasonal_order)
        rerun_is_seasonal = rerun_seasonal and "[0, 0, 0, 0]" not in rerun_seasonal

        action = "no_action"

        if rerun_is_seasonal and ar is not None and abs(ar) > THRESHOLD:
            # attempt to prefer original nonseasonal fit if available
            replaced = False
            if orig and orig.get("series_models") and sid in orig.get("series_models"):
                orig_entry = orig["series_models"].get(sid)
                if orig_entry and orig_entry.get("seasonal_order") == [0, 0, 0, 0]:
                    out_models[sid] = orig_entry
                    action = "replaced_with_orig_nonseasonal"
                    replaced = True

            if not replaced:
                # modify rerun entry: mark seasonal rejected and set seasonal_order to nonseasonal
                entry = out_models.get(sid, {})
                entry = dict(entry)  # copy
                entry["seasonal_order"] = [0, 0, 0, 0]
                entry["seasonal_rejected"] = True
                entry["seasonal_rejected_reason"] = f"max_abs_arroot={ar} > {THRESHOLD}"
                # clear AIC for seasonal model (since we've rejected it)
                entry["aic"] = entry.get("aic") if entry.get("aic") is not None else None
                out_models[sid] = entry
                action = "marked_nonseasonal"

        summary_rows.append({
            "series_id": sid,
            "n_train": info.get("n_train"),
            "rerun_seasonal": rerun_seasonal,
            "max_abs_arroot": ar,
            "action": action,
            "plot": info.get("plot"),
        })

    # assemble output snapshot
    out_snapshot = dict(rerun)
    out_snapshot["series_models"] = out_models
    out_snapshot.setdefault("metadata", {})["arroot_gate_applied"] = True
    out_snapshot["metadata"]["arroot_gate_date"] = datetime.datetime.utcnow().isoformat()

    # write snapshot
    with OUT_SNAPSHOT.open("w", encoding="utf-8") as f:
        json.dump(out_snapshot, f, ensure_ascii=False, indent=2)

    # write summary CSV
    with SUMMARY_CSV.open("w", encoding="utf-8", newline='') as f:
        fieldnames = ["series_id", "n_train", "rerun_seasonal", "max_abs_arroot", "action", "plot"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print(f"Wrote filtered snapshot: {OUT_SNAPSHOT}")
    print(f"Wrote summary CSV: {SUMMARY_CSV}")


if __name__ == '__main__':
    main()
