import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from __future__ import annotations
from src.pipeline import s3_monthly_ts, run_incremental_pipeline
from src.config_loader import load_config

print("OK: imports")
cfg = load_config("configs/pipeline.yaml")
assert hasattr(cfg, "metrics_dir") and hasattr(cfg, "forecasts_dir")
print("OK: config schema")
