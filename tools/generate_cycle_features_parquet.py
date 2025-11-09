"""
Generate cycle_features.parquet from all JSON feature files in data/features/
"""
import os
import json
import pandas as pd
from pathlib import Path

FEATURES_DIR = Path("data/features")
OUTPUT_PATH = FEATURES_DIR / "cycle_features.parquet"

def extract_features_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    series_id = data.get("series_id", "")
    plant = data.get("plant", "")
    product_cat2 = data.get("product_cat2", "")
    mid_category = data.get("mid_category", "")
    records = data.get("data", [])
    # Flatten each record with series info
    for rec in records:
        rec_flat = dict(rec)
        rec_flat["series_id"] = series_id
        rec_flat["plant"] = plant
        rec_flat["product_cat2"] = product_cat2
        rec_flat["mid_category"] = mid_category
        yield rec_flat

def main():
    all_records = []
    for fname in os.listdir(FEATURES_DIR):
        if fname.endswith(".json"):
            json_path = FEATURES_DIR / fname
            for rec in extract_features_from_json(json_path):
                all_records.append(rec)
    if not all_records:
        print("No feature records found.")
        return
    df = pd.DataFrame(all_records)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✅ Saved: {OUTPUT_PATH} ({len(df)} rows)")

if __name__ == "__main__":
    main()
