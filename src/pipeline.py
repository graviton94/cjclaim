
"""CJ Quality-Cycles — Monthly incremental pipeline."""
from __future__ import annotations
import os, sys, json, traceback
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from . import PipelineConfig

def s3_monthly_ts(cfg: 'PipelineConfig', year: int, month: int) -> Dict[str, Any]:
    fmt = _fmt(year, month)
    from scripts.build_monthly_timeseries import build_monthly_timeseries
    curated_parquet = str(cfg.p("curated_parquet", **fmt))
    monthly_ts_path = str(cfg.p("monthly_ts", **fmt))
    return build_monthly_timeseries(curated_parquet, monthly_ts_path)

__all__ = [
    "s0_ingest","s1_curate","s2_features","s3_monthly_ts",
    "s4_fit_models","s5_forecast","s6_reconcile","s7_ews",
    "run_incremental_pipeline",
]

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _fmt(year:int, month:int)->Dict[str, Any]:
    ym = f"{year}{month:02d}"
    return {"year":year, "month":month, "yearmonth":ym}

def _write_text(path: Path, text: str):
    _ensure_dir(path.parent); path.write_text(text, encoding="utf-8")


def s0_ingest(cfg: PipelineConfig, year:int, month:int, raw_csv_src:str)->Dict[str, Any]:
    fmt = _fmt(year, month)
    raw_path = cfg.p("raw_csv", **fmt)
    _ensure_dir(raw_path.parent)
    df = pd.read_csv(raw_csv_src) if isinstance(raw_csv_src, str) and raw_csv_src.lower().endswith(".csv") else pd.read_csv(raw_csv_src)
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    return {"raw_csv": str(raw_path), "rows": len(df)}

def s1_curate(cfg: PipelineConfig, year:int, month:int)->Dict[str, Any]:
    fmt = _fmt(year, month)
    raw_csv = cfg.p("raw_csv", **fmt)
    curated_parquet = cfg.p("curated_parquet", **fmt)
    _ensure_dir(curated_parquet.parent)
    df = pd.read_csv(raw_csv, encoding="utf-8")
    missing = [c for c in cfg.required_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    # 1) 날짜 정규화
    for c in ["발생일자", "제조일자"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["발생일자", "제조일자"])

    # 표준 스키마 생성 및 그룹화
    # 1) series_id 생성
    df["series_id"] = (
        df["플랜트"].astype(str) + "_" +
        df["제품범주2"].astype(str) + "_" +
        df["중분류"].astype(str)
    )
    # 2) 월 키 생성
    date_col = "event_date" if "event_date" in df.columns else "발생일자"
    df["year"]  = df[date_col].dt.year.astype(int)
    df["month"] = df[date_col].dt.month.astype(int)
    df["yyyymm"] = df["year"].astype(str) + df["month"].astype(str).str.zfill(2)
    # 3) claim_count 확정
    if "claim_count" not in df.columns:
        if "count" in df.columns:
            df["claim_count"] = df["count"]
        elif "발생건수" in df.columns:
            df["claim_count"] = df["발생건수"]
        else:
            df["claim_count"] = 1
    # 4) 그룹화 및 저장
    gb_cols = ["series_id", "year", "month", "yyyymm"]
    df = df.groupby(gb_cols, as_index=False).agg(claim_count=("claim_count", "sum"))
    # 5) 저장
    df.to_parquet(curated_parquet, index=False)
    return {"curated_parquet": str(curated_parquet), "rows": len(df)}

def s2_features(cfg: PipelineConfig, year:int, month:int)->Dict[str, Any]:
    fmt = _fmt(year, month)
    from src.cycle_features import extract_series_features
    curated_parquet = cfg.p("curated_parquet", **fmt)
    features_dir = str(cfg.p("features_dir", **fmt))
    updated_list = str(cfg.p("updated_series", **fmt))
    df_curated = pd.read_parquet(curated_parquet)
    # 방어로직: 표준 스키마 보정
    if 'series_id' not in df_curated.columns and all(c in df_curated.columns for c in ["plant","product_category2","mid_category"]):
        df_curated["series_id"] = (
            df_curated["plant"].astype(str) + "_" +
            df_curated["product_category2"].astype(str) + "_" +
            df_curated["mid_category"].astype(str)
        )
    if 'claim_count' not in df_curated.columns:
        if 'count' in df_curated.columns:
            df_curated['claim_count'] = df_curated['count']
        elif '발생건수' in df_curated.columns:
            df_curated['claim_count'] = df_curated['발생건수']
        else:
            df_curated['claim_count'] = 1
    if 'yyyymm' not in df_curated.columns:
        if 'year' in df_curated.columns and 'month' in df_curated.columns:
            df_curated['yyyymm'] = df_curated['year'].astype(str) + df_curated['month'].astype(str).str.zfill(2)
    if 'year' not in df_curated.columns or 'month' not in df_curated.columns:
        if 'yyyymm' in df_curated.columns:
            df_curated['year'] = df_curated['yyyymm'].astype(str).str[:4].astype(int)
            df_curated['month'] = df_curated['yyyymm'].astype(str).str[4:6].astype(int)
    if 'series_id' in df_curated.columns:
        all_series = df_curated['series_id'].unique()
        results = {}
        for series_id in all_series:
            df_series = df_curated[df_curated['series_id'] == series_id]
            lookback_months = min(len(df_series), cfg.opt('lookback_months') or 24)
            res = extract_series_features(
                df_series,
                year,
                month,
                features_dir,
                updated_list
            )
            results[series_id] = res
        return {"series_count": len(all_series), "results": results}
    else:
        raise ValueError("series_id missing in curated data; fix s1_curate output schema.")

def s3_weekly_ts(cfg: PipelineConfig, year:int, month:int)->Dict[str, Any]:
    # [삭제됨] weekly 분석 단계는 더 이상 사용하지 않음
    return {"skipped": True}

def s4_fit_models(cfg: PipelineConfig, year:int, month:int)->Dict[str, Any]:
    fmt = _fmt(year, month)
    from src.forecasting import train_incremental
    updated_series_path = str(cfg.p("updated_series", **fmt))
    monthly_ts_path = str(cfg.p("monthly_ts", **fmt))
    models_dir = "artifacts/models/"
    # Read updated series list
    with open(updated_series_path, "r", encoding="utf-8") as f:
        updated_series_list = [line.strip() for line in f if line.strip()]
    result = train_incremental(updated_series_list, monthly_ts_path, models_dir, warm_start=True, max_workers=4)
    return result

def s5_forecast(cfg: PipelineConfig, year:int, month:int)->Dict[str, Any]:
    fmt = _fmt(year, month)
    from src.forecasting import generate_monthly_forecast
    models_dir = "artifacts/models/"
    horizon_months = cfg.opt("horizon_months") or 12
    forecast_path = str(cfg.p("forecast_file", **fmt))
    monthly_ts_path = str(cfg.p("monthly_ts", **fmt))
    result = generate_monthly_forecast(models_dir, horizon_months, monthly_ts_path, forecast_path)
    return result

def s6_reconcile(cfg: PipelineConfig, year:int, month:int)->Dict[str, Any]:
    fmt = _fmt(year, month)
    from src.reconcile import reconcile_month
    forecast_path = str(cfg.p("forecast_file", **fmt))
    monthly_ts_path = str(cfg.p("monthly_ts", **fmt))
    reconcile_path = str(cfg.p("reconcile", **fmt))
    result = reconcile_month(forecast_path, monthly_ts_path, reconcile_path)
    return result

def s7_ews(cfg: PipelineConfig, year:int, month:int)->Dict[str, Any]:
    fmt = _fmt(year, month)
    from src.ews_scoring_v2 import generate_ews_report
    forecast_path = str(cfg.p("forecast_file", **fmt))
    json_dir = str(cfg.p("features_dir", **fmt))
    metadata_path = str(cfg.p("metadata", **fmt)) if cfg.opt("metadata") else "artifacts/models/base_monthly/training_results.csv"
    ews_path = str(cfg.p("ews_scores", **fmt))
    threshold_path = str(cfg.p("threshold", **fmt)) if cfg.opt("threshold") else "artifacts/metrics/threshold.json"
    top_n = cfg.opt("top_n") or 10
    learn_weights = bool(cfg.opt("learn_weights"))
    df_ews = generate_ews_report(
        forecast_parquet_path=forecast_path,
        json_dir=json_dir,
        metadata_path=metadata_path,
        output_path=ews_path,
        threshold_path=threshold_path,
        top_n=top_n,
        learn_weights=learn_weights
    )
    return {"ews_scores": ews_path, "rows": len(df_ews)}

def run_incremental_pipeline(year:int, month:int, raw_csv_path:str, config_path:str="configs/pipeline.yaml") -> Dict[str, Any]:
    from tools.merge_and_lag_policy import run_merge_and_lag_policy

    from .config_loader import load_config
    cfg = load_config(config_path)
    fmt = _fmt(year, month)
    log_path = cfg.p("log_file", **fmt)
    _ensure_dir(log_path.parent)

    # Lag Policy 자동 생성 루틴 (S0 직후)
    base_candidates = Path("artifacts/metrics/candidates_claims_2021_2023_with_lag_policy.csv")
    if not base_candidates.exists():
        print("기초 Lag Policy 데이터 미존재 → 자동 생성 중...")
        run_merge_and_lag_policy()

    status = {"steps":[], "artifacts_index":{}}
    def _record(name, ok, payload):
        entry = {"name": name, "ok": ok}
        entry.update(payload if isinstance(payload, dict) else {"detail": payload})
        status["steps"].append(entry)
        if ok and isinstance(payload, dict):
            status["artifacts_index"].update(payload)

    try:
        print("[S0] Ingesting raw data...")
        _record("S0_ingest", True, s0_ingest(cfg, year, month, raw_csv_path))
        print("[S0] Done.")
        if cfg.opt("use_curated"):
            print("[S1] Curating data...")
            _record("S1_curate", True, s1_curate(cfg, year, month))
            print("[S1] Done.")
        print("[S2] Extracting features...")
        _record("S2_features", True, s2_features(cfg, year, month))
        print("[S2] Done.")
        print("[S3] Building monthly timeseries...")
        _record("S3_monthly_ts", True, s3_monthly_ts(cfg, year, month))
        print("[S3] Done.")
        print("[S4] Fitting models...")
        _record("S4_fit", True, s4_fit_models(cfg, year, month))
        print("[S4] Done.")
        print("[S5] Forecasting...")
        _record("S5_forecast", True, s5_forecast(cfg, year, month))
        print("[S5] Done.")
        if cfg.opt("enable_reconcile"):
            print("[S6] Reconciling forecasts...")
            _record("S6_reconcile", True, s6_reconcile(cfg, year, month))
            print("[S6] Done.")
        if cfg.opt("enable_ews"):
            print("[S7] Calculating EWS scores...")
            _record("S7_ews", True, s7_ews(cfg, year, month))
            print("[S7] Done.")
    except Exception as e:
        print(f"[ERROR] Pipeline failed at step: {status['steps'][-1]['name'] if status['steps'] else 'init'}")
        print(f"[ERROR] {e}")
        tb = traceback.format_exc()
        _write_text(log_path, tb)
        _record("ERROR", False, {"error": str(e)})
        raise
    return status
