"""
Baseline 학습 결과 검증 도구

학습 직후 모델의 성능을 진단하고 다음을 검증:
1. 잔차 진단 (Ljung-Box, ACF, 정규성)
2. 기준 지표 (MAPE, MASE, Bias)
3. 폴백 모델 사용률

실행 예시:
    python tools/validate_baseline.py --year 2024
    python tools/validate_baseline.py --artifacts artifacts/
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.metrics import (
    compute_all_metrics, 
    compute_metrics_by_group,
    identify_poor_performers
)

warnings.filterwarnings('ignore')


def load_forecast_results(artifacts_path: Path, year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    예측 결과 및 실측값 로드
    
    Args:
        artifacts_path: 아티팩트 디렉토리
        year: 검증 연도
    
    Returns:
        (forecast_df, actual_df) 튜플
    """
    forecast_path = artifacts_path / 'forecasts' / f'forecast_{year}.parquet'
    
    if not forecast_path.exists():
        raise FileNotFoundError(f"예측 결과를 찾을 수 없습니다: {forecast_path}")
    
    forecast_df = pd.read_parquet(forecast_path)
    
    # 실측값 로드 (curated 데이터에서)
    curated_path = project_root / 'data' / 'curated' / f'curated_{year}.parquet'
    if curated_path.exists():
        actual_df = pd.read_parquet(curated_path)
    else:
        print(f"⚠️  실측값이 없습니다: {curated_path}")
        actual_df = None
    
    return forecast_df, actual_df


def analyze_residuals(forecast_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    """
    잔차 분석: Ljung-Box 테스트, ACF, 정규성
    
    Args:
        forecast_df: 예측 데이터프레임
        actual_df: 실측 데이터프레임
    
    Returns:
        잔차 분석 결과 데이터프레임
    """
    from scipy import stats
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    results = []
    
    # 시리즈별 분석
    for series_id in forecast_df['series_id'].unique():
        series_forecast = forecast_df[forecast_df['series_id'] == series_id].copy()
        
        if actual_df is None:
            continue
        
        series_actual = actual_df[actual_df['series_id'] == series_id].copy()
        
        # 기간 매칭
        merged = pd.merge(
            series_forecast[['week_end_date', 'yhat']],
            series_actual[['week_end_date', 'y']],
            on='week_end_date',
            how='inner'
        )
        
        if len(merged) < 10:
            continue
        
        # 잔차 계산
        residuals = merged['y'] - merged['yhat']
        
        # 1. Ljung-Box 테스트 (잔차의 자기상관)
        try:
            lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
            lb_pvalue = lb_test['lb_pvalue'].values[0]
        except:
            lb_pvalue = np.nan
        
        # 2. 정규성 테스트 (Shapiro-Wilk)
        try:
            sw_stat, sw_pvalue = stats.shapiro(residuals)
        except:
            sw_stat, sw_pvalue = np.nan, np.nan
        
        # 3. 기본 통계
        residual_mean = residuals.mean()
        residual_std = residuals.std()
        residual_skew = stats.skew(residuals)
        residual_kurt = stats.kurtosis(residuals)
        
        results.append({
            'series_id': series_id,
            'n_residuals': len(residuals),
            'ljungbox_pvalue': lb_pvalue,
            'ljungbox_pass': lb_pvalue > 0.05 if not np.isnan(lb_pvalue) else None,
            'shapiro_pvalue': sw_pvalue,
            'normality_pass': sw_pvalue > 0.05 if not np.isnan(sw_pvalue) else None,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'residual_skew': residual_skew,
            'residual_kurtosis': residual_kurt,
        })
    
    return pd.DataFrame(results)


def compute_baseline_metrics(forecast_df: pd.DataFrame, 
                            actual_df: pd.DataFrame,
                            train_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    기준선 메트릭 계산
    
    Args:
        forecast_df: 예측 데이터프레임
        actual_df: 실측 데이터프레임
        train_df: 학습 데이터프레임 (MASE용)
    
    Returns:
        메트릭 데이터프레임
    """
    if actual_df is None:
        print("⚠️  실측값이 없어 메트릭을 계산할 수 없습니다.")
        return pd.DataFrame()
    
    # 예측-실측 병합
    merged = pd.merge(
        forecast_df[['series_id', 'week_end_date', 'yhat', 'model_type']],
        actual_df[['series_id', 'week_end_date', 'y']],
        on=['series_id', 'week_end_date'],
        how='inner'
    )
    
    if len(merged) == 0:
        print("⚠️  매칭되는 데이터가 없습니다.")
        return pd.DataFrame()
    
    # 메트릭 계산
    metrics_df = compute_metrics_by_group(
        df=merged,
        y_true_col='y',
        y_pred_col='yhat',
        group_cols=['series_id'],
        y_train=train_df,
        train_group_col='series_id'
    )
    
    # 모델 타입 추가
    model_types = merged.groupby('series_id')['model_type'].first()
    metrics_df = metrics_df.merge(
        model_types.reset_index(),
        on='series_id',
        how='left'
    )
    
    return metrics_df


def analyze_fallback_rate(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    폴백 모델 사용률 분석
    
    Args:
        forecast_df: 예측 데이터프레임
    
    Returns:
        폴백 사용률 데이터프레임
    """
    if 'model_type' not in forecast_df.columns:
        print("⚠️  model_type 컬럼이 없습니다.")
        return pd.DataFrame()
    
    # 시리즈별 모델 타입 집계
    fallback_summary = forecast_df.groupby('series_id').agg({
        'model_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
    }).reset_index()
    
    fallback_summary.columns = ['series_id', 'primary_model']
    
    # 전체 통계
    total_series = len(fallback_summary)
    fallback_counts = fallback_summary['primary_model'].value_counts()
    
    fallback_stats = pd.DataFrame({
        'model_type': fallback_counts.index,
        'count': fallback_counts.values,
        'percentage': (fallback_counts.values / total_series * 100).round(2)
    })
    
    return fallback_summary, fallback_stats


def generate_report(residual_df: pd.DataFrame,
                   metrics_df: pd.DataFrame,
                   fallback_summary: pd.DataFrame,
                   fallback_stats: pd.DataFrame,
                   output_path: Path,
                   year: int):
    """
    종합 보고서 생성
    
    Args:
        residual_df: 잔차 분석 결과
        metrics_df: 메트릭 결과
        fallback_summary: 폴백 요약
        fallback_stats: 폴백 통계
        output_path: 출력 경로
        year: 검증 연도
    """
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / f'baseline_report_{year}.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Baseline 검증 보고서 - {year}년\n\n")
        f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # 1. 전체 요약
        f.write("## 📊 전체 요약\n\n")
        if not metrics_df.empty:
            f.write(f"- **총 시리즈 수**: {len(metrics_df)}\n")
            f.write(f"- **평균 MAPE**: {metrics_df['mape'].mean():.4f}\n")
            f.write(f"- **평균 Bias**: {metrics_df['bias'].mean():.4f}\n")
            if 'mase' in metrics_df.columns:
                f.write(f"- **평균 MASE**: {metrics_df['mase'].mean():.4f}\n")
            f.write(f"- **평균 MAE**: {metrics_df['mae'].mean():.2f}\n")
            f.write(f"- **평균 RMSE**: {metrics_df['rmse'].mean():.2f}\n\n")
        
        # 2. 폴백 모델 통계
        f.write("## 🔄 폴백 모델 사용률\n\n")
        if not fallback_stats.empty:
            f.write("| 모델 타입 | 시리즈 수 | 비율 (%) |\n")
            f.write("|----------|----------|----------|\n")
            for _, row in fallback_stats.iterrows():
                f.write(f"| {row['model_type']} | {row['count']} | {row['percentage']:.2f}% |\n")
            f.write("\n")
        
        # 3. 잔차 진단
        f.write("## 🔍 잔차 진단\n\n")
        if not residual_df.empty:
            ljung_pass = residual_df['ljungbox_pass'].sum()
            ljung_total = residual_df['ljungbox_pass'].notna().sum()
            norm_pass = residual_df['normality_pass'].sum()
            norm_total = residual_df['normality_pass'].notna().sum()
            
            f.write(f"- **Ljung-Box 테스트 통과율**: {ljung_pass}/{ljung_total} ({ljung_pass/ljung_total*100:.1f}%)\n")
            f.write(f"- **정규성 테스트 통과율**: {norm_pass}/{norm_total} ({norm_pass/norm_total*100:.1f}%)\n\n")
            
            # 문제 시리즈
            failed = residual_df[
                (residual_df['ljungbox_pass'] == False) | 
                (residual_df['normality_pass'] == False)
            ]
            
            if len(failed) > 0:
                f.write("### ⚠️ 잔차 테스트 실패 시리즈\n\n")
                f.write(f"총 {len(failed)}개 시리즈\n\n")
        
        # 4. 성능 문제 시리즈
        f.write("## 🎯 튜닝 후보 시리즈\n\n")
        if not metrics_df.empty:
            candidates = identify_poor_performers(
                metrics_df,
                mape_threshold=0.20,
                bias_threshold=0.05,
                mase_threshold=1.5
            )
            
            f.write(f"**MAPE>0.20 또는 Bias>0.05 또는 MASE>1.5 시리즈**: {len(candidates)}개\n\n")
            
            if len(candidates) > 0:
                f.write("### Top 20 우선순위 시리즈\n\n")
                f.write("| Series ID | MAPE | Bias | MASE | 우선순위 |\n")
                f.write("|-----------|------|------|------|----------|\n")
                
                for _, row in candidates.head(20).iterrows():
                    mase_val = f"{row['mase']:.3f}" if 'mase' in row and not pd.isna(row['mase']) else 'N/A'
                    f.write(f"| {row['series_id']} | {row['mape']:.4f} | {row['bias']:.4f} | {mase_val} | {row['priority_score']:.2f} |\n")
                
                f.write("\n")
                
                # 튜닝 후보 CSV 저장
                candidates_path = output_path.parent / 'metrics' / f'tuning_candidates_{year}.csv'
                candidates_path.parent.mkdir(parents=True, exist_ok=True)
                candidates.to_csv(candidates_path, index=False)
                f.write(f"📁 전체 튜닝 후보 목록: `{candidates_path.relative_to(project_root)}`\n\n")
        
        # 5. 모델 타입별 성능
        f.write("## 📈 모델 타입별 성능\n\n")
        if not metrics_df.empty and 'model_type' in metrics_df.columns:
            model_perf = metrics_df.groupby('model_type').agg({
                'mape': 'mean',
                'bias': 'mean',
                'mae': 'mean',
                'series_id': 'count'
            }).round(4)
            model_perf.columns = ['평균_MAPE', '평균_Bias', '평균_MAE', '시리즈_수']
            
            f.write(model_perf.to_markdown())
            f.write("\n\n")
        
        f.write("---\n\n")
        f.write("다음 단계: Rolling 백테스트로 기준선 확립\n")
        f.write("```bash\n")
        f.write("python batch.py roll --start 2020 --end 2024\n")
        f.write("```\n")
    
    print(f"\n✅ 보고서 생성 완료: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Baseline 학습 결과 검증')
    parser.add_argument('--year', type=int, default=2024, help='검증 연도')
    parser.add_argument('--artifacts', type=str, default='artifacts', help='아티팩트 디렉토리')
    parser.add_argument('--output', type=str, default='reports', help='보고서 출력 디렉토리')
    
    args = parser.parse_args()
    
    artifacts_path = Path(args.artifacts)
    output_path = Path(args.output)
    
    print(f"\n{'='*60}")
    print(f"Baseline 검증 - {args.year}년")
    print(f"{'='*60}\n")
    
    # 1. 데이터 로드
    print("📂 데이터 로드 중...")
    forecast_df, actual_df = load_forecast_results(artifacts_path, args.year)
    print(f"   예측 데이터: {len(forecast_df)} 행")
    if actual_df is not None:
        print(f"   실측 데이터: {len(actual_df)} 행")
    
    # 학습 데이터 로드 (MASE 계산용)
    train_path = project_root / 'data' / 'curated' / f'curated_{args.year - 1}.parquet'
    train_df = pd.read_parquet(train_path) if train_path.exists() else None
    
    # 2. 잔차 분석
    print("\n🔍 잔차 분석 중...")
    residual_df = analyze_residuals(forecast_df, actual_df)
    print(f"   분석 완료: {len(residual_df)} 시리즈")
    
    # 3. 메트릭 계산
    print("\n📊 메트릭 계산 중...")
    metrics_df = compute_baseline_metrics(forecast_df, actual_df, train_df)
    print(f"   계산 완료: {len(metrics_df)} 시리즈")
    
    # 4. 폴백 분석
    print("\n🔄 폴백 모델 분석 중...")
    fallback_summary, fallback_stats = analyze_fallback_rate(forecast_df)
    print(f"   분석 완료")
    
    # 5. 메트릭 저장
    print("\n💾 메트릭 저장 중...")
    metrics_path = artifacts_path / 'metrics'
    metrics_path.mkdir(parents=True, exist_ok=True)
    
    if not metrics_df.empty:
        metrics_file = metrics_path / f'metrics_baseline_{args.year}.parquet'
        metrics_df.to_parquet(metrics_file, index=False)
        print(f"   ✓ {metrics_file}")
    
    if not residual_df.empty:
        residual_file = metrics_path / f'residual_analysis_{args.year}.parquet'
        residual_df.to_parquet(residual_file, index=False)
        print(f"   ✓ {residual_file}")
    
    if not fallback_summary.empty:
        fallback_file = metrics_path / f'fallback_summary_{args.year}.csv'
        fallback_summary.to_csv(fallback_file, index=False)
        print(f"   ✓ {fallback_file}")
    
    # 6. 보고서 생성
    print("\n📝 보고서 생성 중...")
    generate_report(
        residual_df, metrics_df, fallback_summary, fallback_stats,
        output_path, args.year
    )
    
    # 7. 요약 출력
    print(f"\n{'='*60}")
    print("요약")
    print(f"{'='*60}")
    
    if not metrics_df.empty:
        print(f"\n총 시리즈: {len(metrics_df)}")
        print(f"평균 MAPE: {metrics_df['mape'].mean():.4f}")
        print(f"평균 Bias: {metrics_df['bias'].mean():.4f}")
        
        candidates = identify_poor_performers(metrics_df)
        print(f"\n튜닝 후보 시리즈: {len(candidates)}개 ({len(candidates)/len(metrics_df)*100:.1f}%)")
    
    if not fallback_stats.empty:
        print("\n폴백 모델 사용률:")
        for _, row in fallback_stats.iterrows():
            print(f"  {row['model_type']}: {row['percentage']:.1f}%")
    
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
