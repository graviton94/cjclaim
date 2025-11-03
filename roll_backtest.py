"""
Rolling 백테스트 파이프라인

연도별로 순차적으로 학습-예측-평가를 반복하여
모델의 시간에 따른 성능을 측정하고 튜닝 후보를 선별

실행 예시:
    python roll_backtest.py --start 2020 --end 2024
    python roll_backtest.py --start 2020 --end 2024 --series "series_123"
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import warnings

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline_train import train_until
from pipeline_forecast import forecast_year
from src.metrics import compute_metrics_by_group, identify_poor_performers

warnings.filterwarnings('ignore')


def run_rolling_backtest(curated_path: Path,
                         start_year: int,
                         end_year: int,
                         series_filter: str = "all",
                         engine: str = "pandas") -> pd.DataFrame:
    """
    Rolling 백테스트 실행
    
    각 연도에 대해:
    1. 해당 연도까지의 데이터로 모델 학습
    2. 다음 연도 예측
    3. 실측값과 비교하여 메트릭 계산
    
    Args:
        curated_path: Curated 데이터 경로
        start_year: 시작 연도
        end_year: 종료 연도 (이 연도까지 학습, 다음 연도 예측)
        series_filter: 시리즈 필터
        engine: 실행 엔진
    
    Returns:
        연도별 메트릭 데이터프레임
    """
    all_metrics = []
    
    print(f"\n{'='*70}")
    print(f"Rolling 백테스트: {start_year} ~ {end_year}")
    print(f"{'='*70}\n")
    
    # 전체 데이터 로드
    if curated_path.exists():
        full_data = pd.read_parquet(curated_path)
    else:
        raise FileNotFoundError(f"Curated 데이터를 찾을 수 없습니다: {curated_path}")
    
    for train_year in range(start_year, end_year + 1):
        test_year = train_year + 1
        
        print(f"\n{'='*70}")
        print(f"[{train_year}년 학습 → {test_year}년 예측]")
        print(f"{'='*70}\n")
        
        # 1. 학습
        print(f"📚 {train_year}년까지 데이터로 학습 중...")
        try:
            train_until(curated_path, train_year)
            print(f"   ✓ 학습 완료")
        except Exception as e:
            print(f"   ✗ 학습 실패: {e}")
            continue
        
        # 2. 예측
        print(f"\n🔮 {test_year}년 예측 중...")
        try:
            forecast_year(curated_path, test_year)
            print(f"   ✓ 예측 완료")
        except Exception as e:
            print(f"   ✗ 예측 실패: {e}")
            continue
        
        # 3. 평가
        print(f"\n📊 {test_year}년 실측값과 비교 중...")
        
        # 예측 결과 로드
        forecast_path = Path('artifacts/forecasts') / f'forecast_{test_year}.parquet'
        if not forecast_path.exists():
            print(f"   ⚠️  예측 결과가 없습니다: {forecast_path}")
            continue
        
        forecast_df = pd.read_parquet(forecast_path)
        
        # 실측값 필터링
        actual_df = full_data[full_data['year'] == test_year].copy()
        
        if len(actual_df) == 0:
            print(f"   ⚠️  {test_year}년 실측값이 없습니다.")
            continue
        
        # 예측-실측 병합
        merged = pd.merge(
            forecast_df[['series_id', 'week_end_date', 'yhat', 'model_type']],
            actual_df[['series_id', 'week_end_date', 'y']],
            on=['series_id', 'week_end_date'],
            how='inner'
        )
        
        if len(merged) == 0:
            print(f"   ⚠️  매칭되는 데이터가 없습니다.")
            continue
        
        # 학습 데이터 (MASE 계산용)
        train_df = full_data[full_data['year'] <= train_year].copy()
        
        # 메트릭 계산
        year_metrics = compute_metrics_by_group(
            df=merged,
            y_true_col='y',
            y_pred_col='yhat',
            group_cols=['series_id'],
            y_train=train_df,
            train_group_col='series_id'
        )
        
        # 연도 정보 추가
        year_metrics['train_year'] = train_year
        year_metrics['test_year'] = test_year
        
        # 모델 타입 추가
        model_types = merged.groupby('series_id')['model_type'].first()
        year_metrics = year_metrics.merge(
            model_types.reset_index(),
            on='series_id',
            how='left'
        )
        
        all_metrics.append(year_metrics)
        
        print(f"   ✓ 평가 완료: {len(year_metrics)} 시리즈")
        print(f"   - 평균 MAPE: {year_metrics['mape'].mean():.4f}")
        print(f"   - 평균 Bias: {year_metrics['bias'].mean():.4f}")
        if 'mase' in year_metrics.columns:
            print(f"   - 평균 MASE: {year_metrics['mase'].mean():.4f}")
    
    # 전체 결과 병합
    if len(all_metrics) == 0:
        print("\n⚠️  메트릭이 계산되지 않았습니다.")
        return pd.DataFrame()
    
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    
    return all_metrics_df


def analyze_trends(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    시계열별 성능 트렌드 분석
    
    Args:
        metrics_df: Rolling 백테스트 메트릭
    
    Returns:
        트렌드 분석 결과
    """
    if metrics_df.empty:
        return pd.DataFrame()
    
    trends = []
    
    for series_id in metrics_df['series_id'].unique():
        series_data = metrics_df[metrics_df['series_id'] == series_id].sort_values('test_year')
        
        if len(series_data) < 2:
            continue
        
        # 성능 변화 계산
        mape_trend = series_data['mape'].diff().mean()  # 평균 변화율
        bias_trend = series_data['bias'].diff().mean()
        
        # 최근 성능
        recent_mape = series_data['mape'].iloc[-1]
        recent_bias = series_data['bias'].iloc[-1]
        
        # 평균 성능
        avg_mape = series_data['mape'].mean()
        avg_bias = series_data['bias'].mean()
        
        # 성능 악화 여부
        is_degrading = mape_trend > 0.01  # MAPE가 증가 추세
        
        trends.append({
            'series_id': series_id,
            'n_years': len(series_data),
            'avg_mape': avg_mape,
            'avg_bias': avg_bias,
            'recent_mape': recent_mape,
            'recent_bias': recent_bias,
            'mape_trend': mape_trend,
            'bias_trend': bias_trend,
            'is_degrading': is_degrading,
        })
    
    return pd.DataFrame(trends)


def generate_rolling_report(metrics_df: pd.DataFrame,
                            trends_df: pd.DataFrame,
                            start_year: int,
                            end_year: int,
                            output_path: Path):
    """
    Rolling 백테스트 보고서 생성
    
    Args:
        metrics_df: 메트릭 데이터프레임
        trends_df: 트렌드 분석 결과
        start_year: 시작 연도
        end_year: 종료 연도
        output_path: 출력 경로
    """
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / f'rolling_backtest_{start_year}_{end_year}.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Rolling 백테스트 보고서\n\n")
        f.write(f"**기간**: {start_year} ~ {end_year}\n")
        f.write(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # 1. 연도별 전체 성능
        f.write("## 📊 연도별 전체 성능\n\n")
        if not metrics_df.empty:
            yearly_summary = metrics_df.groupby('test_year').agg({
                'mape': 'mean',
                'bias': 'mean',
                'mae': 'mean',
                'series_id': 'count'
            }).round(4)
            yearly_summary.columns = ['평균_MAPE', '평균_Bias', '평균_MAE', '시리즈_수']
            
            f.write(yearly_summary.to_markdown())
            f.write("\n\n")
        
        # 2. 모델 타입별 성능
        f.write("## 🔧 모델 타입별 성능 (전체 기간)\n\n")
        if 'model_type' in metrics_df.columns:
            model_summary = metrics_df.groupby('model_type').agg({
                'mape': 'mean',
                'bias': 'mean',
                'series_id': 'count'
            }).round(4)
            model_summary.columns = ['평균_MAPE', '평균_Bias', '시리즈_수']
            
            f.write(model_summary.to_markdown())
            f.write("\n\n")
        
        # 3. 성능 악화 시리즈
        f.write("## ⚠️ 성능 악화 추세 시리즈\n\n")
        if not trends_df.empty:
            degrading = trends_df[trends_df['is_degrading'] == True].sort_values(
                'mape_trend', ascending=False
            )
            
            f.write(f"**총 {len(degrading)}개 시리즈**에서 성능 악화 추세 감지\n\n")
            
            if len(degrading) > 0:
                f.write("### Top 20 악화 시리즈\n\n")
                f.write("| Series ID | 평균 MAPE | 최근 MAPE | MAPE 증가율 | Bias 변화 |\n")
                f.write("|-----------|-----------|-----------|-------------|----------|\n")
                
                for _, row in degrading.head(20).iterrows():
                    f.write(f"| {row['series_id']} | {row['avg_mape']:.4f} | {row['recent_mape']:.4f} | {row['mape_trend']:+.4f} | {row['bias_trend']:+.4f} |\n")
                
                f.write("\n")
        
        # 4. 튜닝 후보 시리즈 (최종 연도 기준)
        f.write("## 🎯 튜닝 후보 시리즈 (최종 연도 기준)\n\n")
        if not metrics_df.empty:
            latest_year = metrics_df['test_year'].max()
            latest_metrics = metrics_df[metrics_df['test_year'] == latest_year]
            
            candidates = identify_poor_performers(
                latest_metrics,
                mape_threshold=0.20,
                bias_threshold=0.05,
                mase_threshold=1.5
            )
            
            f.write(f"**{latest_year}년 기준**: {len(candidates)}개 시리즈가 튜닝 필요\n\n")
            
            if len(candidates) > 0:
                f.write("### Top 20 우선순위 시리즈\n\n")
                f.write("| Series ID | MAPE | Bias | MASE | 우선순위 |\n")
                f.write("|-----------|------|------|------|----------|\n")
                
                for _, row in candidates.head(20).iterrows():
                    mase_val = f"{row['mase']:.3f}" if 'mase' in row and not pd.isna(row['mase']) else 'N/A'
                    f.write(f"| {row['series_id']} | {row['mape']:.4f} | {row['bias']:.4f} | {mase_val} | {row['priority_score']:.2f} |\n")
                
                f.write("\n")
                
                # 튜닝 후보 CSV 저장
                candidates_path = Path('artifacts/metrics') / f'tuning_candidates_rolling_{start_year}_{end_year}.csv'
                candidates_path.parent.mkdir(parents=True, exist_ok=True)
                candidates.to_csv(candidates_path, index=False)
                f.write(f"📁 전체 튜닝 후보 목록: `{candidates_path}`\n\n")
        
        # 5. 일관성 분석
        f.write("## 📈 시리즈 일관성 분석\n\n")
        if not trends_df.empty:
            # MAPE 표준편차가 높은 시리즈 = 예측이 불안정
            unstable = metrics_df.groupby('series_id')['mape'].std().reset_index()
            unstable.columns = ['series_id', 'mape_std']
            unstable = unstable.sort_values('mape_std', ascending=False)
            
            f.write("### 예측 불안정 시리즈 (MAPE 편차 높음)\n\n")
            f.write("| Series ID | MAPE 표준편차 |\n")
            f.write("|-----------|---------------|\n")
            
            for _, row in unstable.head(10).iterrows():
                f.write(f"| {row['series_id']} | {row['mape_std']:.4f} |\n")
            
            f.write("\n")
        
        f.write("---\n\n")
        f.write("## 다음 단계\n\n")
        f.write("1. **경량 보정 적용**: Bias Map, Seasonal Recalibration\n")
        f.write("2. **Optuna 튜닝**: 상위 우선순위 시리즈부터\n")
        f.write("3. **재평가**: 개선 효과 측정\n\n")
        f.write("```bash\n")
        f.write("# 보정 파이프라인 실행\n")
        f.write(f"python batch.py reconcile --year {end_year + 1} --kpi-mape 0.20\n\n")
        f.write("# Optuna 튜닝\n")
        f.write(f"python tools/run_optuna.py --candidates artifacts/metrics/tuning_candidates_rolling_{start_year}_{end_year}.csv\n")
        f.write("```\n")
    
    print(f"\n✅ 보고서 생성 완료: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Rolling 백테스트 실행')
    parser.add_argument('--start', type=int, required=True, help='시작 연도')
    parser.add_argument('--end', type=int, required=True, help='종료 연도 (학습)')
    parser.add_argument('--series', type=str, default='all', help='시리즈 필터')
    parser.add_argument('--engine', type=str, default='pandas', help='실행 엔진')
    parser.add_argument('--curated', type=str, default='data/curated/claims.parquet', help='Curated 데이터 경로')
    parser.add_argument('--output', type=str, default='reports', help='보고서 출력 디렉토리')
    
    args = parser.parse_args()
    
    curated_path = Path(args.curated)
    output_path = Path(args.output)
    
    # 1. Rolling 백테스트 실행
    metrics_df = run_rolling_backtest(
        curated_path=curated_path,
        start_year=args.start,
        end_year=args.end,
        series_filter=args.series,
        engine=args.engine
    )
    
    if metrics_df.empty:
        print("\n⚠️  메트릭이 계산되지 않았습니다. 종료합니다.")
        return
    
    # 2. 트렌드 분석
    print(f"\n{'='*70}")
    print("트렌드 분석 중...")
    print(f"{'='*70}\n")
    
    trends_df = analyze_trends(metrics_df)
    
    # 3. 결과 저장
    print("\n💾 결과 저장 중...")
    
    metrics_path = Path('artifacts/metrics')
    metrics_path.mkdir(parents=True, exist_ok=True)
    
    # 메트릭 저장
    metrics_file = metrics_path / f'rolling_metrics_{args.start}_{args.end}.parquet'
    metrics_df.to_parquet(metrics_file, index=False)
    print(f"   ✓ {metrics_file}")
    
    # 트렌드 저장
    if not trends_df.empty:
        trends_file = metrics_path / f'rolling_trends_{args.start}_{args.end}.parquet'
        trends_df.to_parquet(trends_file, index=False)
        print(f"   ✓ {trends_file}")
    
    # 4. 보고서 생성
    print("\n📝 보고서 생성 중...")
    generate_rolling_report(
        metrics_df=metrics_df,
        trends_df=trends_df,
        start_year=args.start,
        end_year=args.end,
        output_path=output_path
    )
    
    # 5. 요약 출력
    print(f"\n{'='*70}")
    print("Rolling 백테스트 완료")
    print(f"{'='*70}\n")
    
    print(f"총 테스트 기간: {args.start} ~ {args.end + 1}")
    print(f"총 시리즈: {metrics_df['series_id'].nunique()}")
    print(f"총 관측치: {len(metrics_df)}")
    
    print(f"\n전체 평균 성능:")
    print(f"  MAPE: {metrics_df['mape'].mean():.4f}")
    print(f"  Bias: {metrics_df['bias'].mean():.4f}")
    print(f"  MAE: {metrics_df['mae'].mean():.2f}")
    
    if not trends_df.empty:
        degrading_count = trends_df['is_degrading'].sum()
        print(f"\n성능 악화 시리즈: {degrading_count}개 ({degrading_count/len(trends_df)*100:.1f}%)")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
