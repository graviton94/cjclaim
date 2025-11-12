from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PipelineConfig:
    def required_columns(self):
        import yaml
        cfg_path = Path("configs/pipeline.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        return list(cfg.get("schema", {}).get("required_columns", []))
    def opt(self, key: str):
        import yaml
        cfg_path = Path("configs/pipeline.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        return cfg.get("options", {}).get(key)
    raw_dir: Path
    curated_dir: Path
    features_dir: Path
    metrics_dir: Path
    forecasts_dir: Path
    models_dir: Path
    temp_dir: Path
    horizon_months: int = 6

    def p(self, key: str, **kwargs) -> Path:
        # pipeline.yaml의 paths에서 key를 찾아 포맷팅
        import yaml
        cfg_path = Path("configs/pipeline.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        s = cfg["paths"][key].format(**kwargs)
        return Path(s)