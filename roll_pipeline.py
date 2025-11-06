"""
2020~2024 연도별 롤링 학습/예측/검증 자동화 스크립트
"""
from pipeline_train import train_until
from pipeline_forecast import forecast_year
from pipeline_reconcile import reconcile_year
from pathlib import Path

CURATED_PATH = "data/curated/claims.parquet"
START_YEAR = 2020
END_YEAR = 2024

for year in range(START_YEAR, END_YEAR):
    print(f"[TRAIN] {year}까지 학습...")
    train_until(CURATED_PATH, year)
    print(f"[FORECAST] {year+1} 예측...")
    forecast_year(CURATED_PATH, year+1)
    print(f"[RECONCILE] {year+1} 검증/보정...")
    reconcile_year(CURATED_PATH, year+1)
    
print("[SUCCESS] 전체 파이프라인 완료.")
print(f"📊 결과 위치: artifacts/forecasts/, artifacts/adjustments/")