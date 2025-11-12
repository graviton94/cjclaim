
# scripts/cli_run_incremental.py — quick CLI for local tests
import argparse, json
from src.pipeline import run_incremental_pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/pipeline.yaml")
    args = ap.parse_args()
    out = run_incremental_pipeline(args.year, args.month, args.csv, args.config)
    print(json.dumps(out, ensure_ascii=False, indent=2))
