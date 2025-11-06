"""
월별 증분학습 파이프라인
1. 월별 데이터 업로드 (발생일자 기준)
2. Lag 필터링 (Normal-Lag만 학습)
3. 기존 예측과 비교
4. 모델 재학습 (append_fit)
5. 결과 기록
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import argparse
import sys

# 로컬 모듈 임포트
from tools.filter_monthly_data import filter_monthly_data
from tools.compare_forecast_actual import compare_forecast_vs_actual

def process_monthly_data(
    upload_file: str,
    year: int,
    month: int,
    lag_stats_path: str = "artifacts/metrics/lag_stats_from_raw.csv",
    base_models_dir: str = "artifacts/models/base_2021_2023",
    output_dir: str = "artifacts/incremental"
):
    """
    월별 데이터 처리 및 증분학습
    
    Parameters:
    -----------
    upload_file : str
        업로드된 월별 데이터 CSV 경로
    year : int
        대상 연도 (예: 2024)
    month : int
        대상 월 (1-12)
    """
    
    print("=" * 80)
    print(f"월별 증분학습 파이프라인: {year}-{month:02d}")
    print("=" * 80)
    
    # Step 1: 업로드 데이터 로드
    print(f"\n[Step 1] 데이터 로드: {upload_file}")
    df_upload = pd.read_csv(upload_file, encoding='utf-8')
    print(f"  원본 레코드: {len(df_upload):,}건")
    
    # Step 2: Lag 필터링
    print(f"\n[Step 2] Lag 필터링 (Normal-Lag만 학습)")
    filter_stats = filter_monthly_data(
        input_csv=upload_file,
        year=year,
        month=month,
        lag_stats_path=lag_stats_path,
        output_dir=output_dir
    )
    
    # Step 3: 기존 예측 로드 및 비교
    print(f"\n[Step 3] 예측-실측 비교")
    forecast_file = f"artifacts/forecasts/{year}/forecast_{year}_{month:02d}.parquet"
    
    if Path(forecast_file).exists():
        series_metrics = compare_forecast_vs_actual(
            actual_file=filter_stats['candidates_file'],
            forecast_file=forecast_file,
            year=year,
            month=month,
            output_dir="logs"
        )
    else:
        print(f"  ⚠️ 예측 파일 없음: {forecast_file}")
        print(f"  → 예측 없이 학습만 진행")
        series_metrics = {}
    
    # Step 4: KPI 게이트 체크
    print(f"\n[Step 4] KPI 게이트 체크")
    kpi_pass = True
    if series_metrics:
        import numpy as np
        valid_mapes = [m['MAPE'] for m in series_metrics.values() if m['MAPE'] is not None]
        valid_bias = [abs(m['Bias']) for m in series_metrics.values() if m['Bias'] is not None]
        
        if valid_mapes:
            avg_mape = np.mean(valid_mapes)
            print(f"  평균 MAPE: {avg_mape:.2f}% (목표: <20%)")
            if avg_mape > 20:
                kpi_pass = False
        
        if valid_bias:
            avg_bias = np.mean(valid_bias)
            print(f"  평균 |Bias|: {avg_bias:.4f} (목표: <0.05)")
            if avg_bias > 0.05:
                kpi_pass = False
        
        if kpi_pass:
            print(f"  ✅ KPI 통과")
        else:
            print(f"  ⚠️ KPI 미달 - Reconcile 필요")
    
    # Step 5: 모델 재학습 (append_fit)
    print(f"\n[Step 5] 모델 재학습 (Append Fit)")
    print(f"  TODO: append_fit 구현 예정")
    print(f"  학습 후보: {filter_stats['normal'] + filter_stats['borderline']:,}건")
    
    # Step 6: 결과 기록
    print(f"\n[Step 6] 결과 저장")
    log_dir = Path("logs/incremental")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'year': year,
        'month': month,
        'timestamp': datetime.now().isoformat(),
        'filter_stats': filter_stats,
        'kpi_pass': kpi_pass,
        'series_count': len(series_metrics),
        'forecast_file': forecast_file,
        'candidates_file': filter_stats['candidates_file']
    }
    
    log_file = log_dir / f"summary_{year}_{month:02d}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ 요약 저장: {log_file}")
    
    print("\n" + "=" * 80)
    print("✅ 월별 증분학습 완료!")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="월별 증분학습 파이프라인")
    parser.add_argument("--upload", type=str, required=True,
                        help="업로드된 월별 데이터 CSV")
    parser.add_argument("--year", type=int, required=True,
                        help="대상 연도 (예: 2024)")
    parser.add_argument("--month", type=int, required=True,
                        help="대상 월 (1-12)")
    args = parser.parse_args()
    
    process_monthly_data(
        upload_file=args.upload,
        year=args.year,
        month=args.month
    )
