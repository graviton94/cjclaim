from __future__ import annotations
from pathlib import Path
import yaml
from .config_models import PipelineConfig

def load_config(yaml_path: Path | str) -> PipelineConfig:
    yaml_path = Path(yaml_path)
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    base = Path(cfg["paths"]["base"]).resolve()
    return PipelineConfig(
        raw_dir=base / cfg["paths"]["raw"],
        curated_dir=base / cfg["paths"]["curated"],
        features_dir=base / cfg["paths"]["features"],
        metrics_dir=base / cfg["paths"]["metrics"],
        forecasts_dir=base / cfg["paths"]["forecasts"],
        models_dir=base / cfg["paths"]["models"],
        temp_dir=base / cfg["paths"]["temp"],
        horizon_months=int(cfg.get("options", {}).get("horizon_months", 6)),
    )
