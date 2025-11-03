"""
Optuna 기반 SARIMAX 하이퍼파라미터 튜닝

성능이 낮은 시리즈에 대해 자동으로 최적의 SARIMAX 파라미터를 탐색

실행 예시:
    python tools/run_optuna.py --candidates artifacts/metrics/tuning_candidates.csv
    python tools/run_optuna.py --candidates artifacts/metrics/tuning_candidates.csv --timeout 600 --n-trials 40
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.metrics import compute_mase, compute_mape

warnings.filterwarnings('ignore')


def objective(trial, y: np.ndarray, seasonal_period: int = 52) -> float:
    """
    Optuna 목표 함수 - SARIMAX 파라미터 최적화
    
    Args:
        trial: Optuna trial 객체
        y: 시계열 데이터
        seasonal_period: 계절 주기
    
    Returns:
        검증 MASE 값
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # 파라미터 탐색 공간
    p = trial.suggest_int('p', 0, 2)
    d = trial.suggest_int('d', 0, 2)
    q = trial.suggest_int('q', 0, 2)
    P = trial.suggest_int('P', 0, 2)
    D = trial.suggest_int('D', 0, 2)
    Q = trial.suggest_int('Q', 0, 2)
    
    # 너무 복잡한 모델 방지
    if (p + d + q + P + D + Q) > 8:
        return float('inf')
    
    try:
        # 학습/검증 분할 (80/20)
        train_size = int(len(y) * 0.8)
        y_train = y[:train_size]
        y_val = y[train_size:]
        
        # 모델 학습
        model = SARIMAX(
            y_train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted = model.fit(disp=False, maxiter=100)
        
        # 검증 예측
        forecast = fitted.forecast(steps=len(y_val))
        
        # MASE 계산
        mase = compute_mase(y_val, forecast, y_train, seasonal_period)
        
        # NaN 체크
        if np.isnan(mase) or np.isinf(mase):
            return float('inf')
        
        return mase
        
    except Exception as e:
        # 모델 학습 실패 시 페널티
        return float('inf')


def tune_single_series(series_id: str,
                      y: np.ndarray,
                      timeout: int = 600,
                      n_trials: int = 40,
                      seasonal_period: int = 52) -> Dict:
    """
    단일 시리즈 튜닝
    
    Args:
        series_id: 시리즈 ID
        y: 시계열 데이터
        timeout: 최대 실행 시간 (초)
        n_trials: 최대 trial 수
        seasonal_period: 계절 주기
    
    Returns:
        튜닝 결과 딕셔너리
    """
    import optuna
    
    print(f"  🔧 튜닝 시작: {series_id}")
    
    # Study 생성
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # 최적화 실행
    try:
        study.optimize(
            lambda trial: objective(trial, y, seasonal_period),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False
        )
        
        # 결과 저장
        result = {
            'series_id': series_id,
            'best_mase': study.best_value,
            'n_trials': len(study.trials),
            'best_params': study.best_params,
            'status': 'success',
        }
        
        print(f"  ✓ 완료: {series_id} - MASE: {study.best_value:.4f}")
        
    except Exception as e:
        result = {
            'series_id': series_id,
            'best_mase': np.nan,
            'n_trials': 0,
            'best_params': {},
            'status': f'failed: {str(e)}',
        }
        print(f"  ✗ 실패: {series_id} - {str(e)}")
    
    return result


def tune_series_parallel(candidates_df: pd.DataFrame,
                         curated_path: Path,
                         timeout: int = 600,
                         n_trials: int = 40,
                         max_workers: int = 6,
                         max_series: int = None) -> pd.DataFrame:
    """
    병렬로 여러 시리즈 튜닝
    
    Args:
        candidates_df: 튜닝 후보 시리즈 데이터프레임
        curated_path: Curated 데이터 경로
        timeout: 시리즈당 최대 시간 (초)
        n_trials: 시리즈당 trial 수
        max_workers: 병렬 작업자 수
        max_series: 최대 튜닝 시리즈 수 (None = 전체)
    
    Returns:
        튜닝 결과 데이터프레임
    """
    # 데이터 로드
    if not curated_path.exists():
        raise FileNotFoundError(f"Curated 데이터를 찾을 수 없습니다: {curated_path}")
    
    full_data = pd.read_parquet(curated_path)
    
    # 우선순위 정렬
    candidates_df = candidates_df.sort_values('priority_score', ascending=False)
    
    if max_series:
        candidates_df = candidates_df.head(max_series)
    
    print(f"\n{'='*70}")
    print(f"Optuna 튜닝 시작: {len(candidates_df)}개 시리즈")
    print(f"병렬 작업자: {max_workers}, Timeout: {timeout}초, Trials: {n_trials}")
    print(f"{'='*70}\n")
    
    # 튜닝 작업 준비
    tasks = []
    for _, row in candidates_df.iterrows():
        series_id = row['series_id']
        series_data = full_data[full_data['series_id'] == series_id]['y'].values
        
        if len(series_data) < 52:
            print(f"  ⚠️  데이터 부족: {series_id} ({len(series_data)}주)")
            continue
        
        tasks.append((series_id, series_data, timeout, n_trials, 52))
    
    # 병렬 실행
    results = []
    
    # 순차 실행 (간단한 버전)
    for series_id, y, timeout, n_trials, period in tasks:
        result = tune_single_series(series_id, y, timeout, n_trials, period)
        results.append(result)
    
    # 병렬 실행 (선택적 - 주석 처리)
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     futures = {
    #         executor.submit(tune_single_series, *task): task[0]
    #         for task in tasks
    #     }
    #     
    #     for future in as_completed(futures):
    #         series_id = futures[future]
    #         try:
    #             result = future.result()
    #             results.append(result)
    #         except Exception as e:
    #             print(f"  ✗ 오류: {series_id} - {e}")
    
    return pd.DataFrame(results)


def save_tuned_params(results_df: pd.DataFrame,
                     output_path: Path,
                     year: int = None):
    """
    튜닝된 파라미터 저장
    
    Args:
        results_df: 튜닝 결과 데이터프레임
        output_path: 출력 디렉토리
        year: 연도 (선택)
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 성공한 튜닝 결과만 필터링
    success_df = results_df[results_df['status'] == 'success'].copy()
    
    if len(success_df) == 0:
        print("⚠️  성공한 튜닝 결과가 없습니다.")
        return
    
    # 파라미터 JSON 변환
    params_list = []
    for _, row in success_df.iterrows():
        params_list.append({
            'series_id': row['series_id'],
            'params': row['best_params'],
            'mase': row['best_mase'],
        })
    
    # JSON 저장
    year_suffix = f"_{year}" if year else ""
    json_file = output_path / f'tuned_params{year_suffix}.json'
    
    with open(json_file, 'w') as f:
        json.dump(params_list, f, indent=2)
    
    print(f"\n✅ 튜닝 파라미터 저장: {json_file}")
    print(f"   성공: {len(success_df)}개 시리즈")


def generate_tuning_report(results_df: pd.DataFrame,
                          candidates_df: pd.DataFrame,
                          output_path: Path,
                          year: int = None):
    """
    튜닝 보고서 생성
    
    Args:
        results_df: 튜닝 결과
        candidates_df: 원본 후보 리스트
        output_path: 출력 경로
        year: 연도 (선택)
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    year_suffix = f"_{year}" if year else ""
    report_file = output_path / f'optuna_tuning_report{year_suffix}.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Optuna 튜닝 보고서\n\n")
        if year:
            f.write(f"**연도**: {year}\n")
        f.write(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # 1. 전체 요약
        f.write("## 📊 전체 요약\n\n")
        f.write(f"- **튜닝 대상**: {len(candidates_df)}개 시리즈\n")
        f.write(f"- **튜닝 완료**: {len(results_df)}개 시리즈\n")
        
        success_count = (results_df['status'] == 'success').sum()
        f.write(f"- **성공**: {success_count}개\n")
        f.write(f"- **실패**: {len(results_df) - success_count}개\n\n")
        
        # 2. 성능 개선
        success_df = results_df[results_df['status'] == 'success'].copy()
        
        if len(success_df) > 0:
            f.write("## 🎯 성능 개선\n\n")
            
            # 후보 리스트와 병합
            merged = success_df.merge(
                candidates_df[['series_id', 'mape', 'mase']],
                on='series_id',
                how='left',
                suffixes=('_tuned', '_original')
            )
            
            if 'mase_original' in merged.columns:
                merged['mase_improvement'] = (
                    (merged['mase_original'] - merged['best_mase']) / merged['mase_original'] * 100
                )
                
                f.write(f"- **평균 MASE 개선**: {merged['mase_improvement'].mean():.2f}%\n")
                f.write(f"- **최대 MASE 개선**: {merged['mase_improvement'].max():.2f}%\n\n")
                
                # Top 개선 시리즈
                f.write("### Top 10 개선 시리즈\n\n")
                f.write("| Series ID | 원본 MASE | 튜닝 후 MASE | 개선율 (%) |\n")
                f.write("|-----------|-----------|--------------|------------|\n")
                
                top_improved = merged.nlargest(10, 'mase_improvement')
                for _, row in top_improved.iterrows():
                    f.write(f"| {row['series_id']} | {row['mase_original']:.4f} | {row['best_mase']:.4f} | {row['mase_improvement']:+.2f}% |\n")
                
                f.write("\n")
        
        # 3. 파라미터 분포
        if len(success_df) > 0:
            f.write("## 🔧 최적 파라미터 분포\n\n")
            
            # 파라미터 추출
            params_df = pd.DataFrame([
                row['best_params'] for _, row in success_df.iterrows()
            ])
            
            f.write("| 파라미터 | 평균 | 최빈값 |\n")
            f.write("|----------|------|--------|\n")
            
            for col in ['p', 'd', 'q', 'P', 'D', 'Q']:
                if col in params_df.columns:
                    mean_val = params_df[col].mean()
                    mode_val = params_df[col].mode()[0] if len(params_df[col].mode()) > 0 else 0
                    f.write(f"| {col} | {mean_val:.2f} | {mode_val} |\n")
            
            f.write("\n")
        
        # 4. 실패 분석
        failed_df = results_df[results_df['status'] != 'success']
        if len(failed_df) > 0:
            f.write("## ⚠️ 튜닝 실패 시리즈\n\n")
            f.write(f"총 {len(failed_df)}개 시리즈\n\n")
            
            f.write("| Series ID | 상태 |\n")
            f.write("|-----------|------|\n")
            
            for _, row in failed_df.head(20).iterrows():
                f.write(f"| {row['series_id']} | {row['status']} |\n")
            
            f.write("\n")
        
        f.write("---\n\n")
        f.write("## 다음 단계\n\n")
        f.write("1. **튜닝된 파라미터 적용**: 재학습 및 재예측\n")
        f.write("2. **성능 재평가**: 실측값과 비교\n")
        f.write("3. **프로덕션 배포**: 검증 후 적용\n\n")
    
    print(f"✅ 보고서 생성 완료: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Optuna 하이퍼파라미터 튜닝')
    parser.add_argument('--candidates', type=str, required=True, help='튜닝 후보 CSV 파일')
    parser.add_argument('--curated', type=str, default='data/curated/claims.parquet', help='Curated 데이터 경로')
    parser.add_argument('--timeout', type=int, default=600, help='시리즈당 최대 시간 (초)')
    parser.add_argument('--n-trials', type=int, default=40, help='시리즈당 trial 수')
    parser.add_argument('--max-workers', type=int, default=6, help='병렬 작업자 수')
    parser.add_argument('--max-series', type=int, default=None, help='최대 튜닝 시리즈 수')
    parser.add_argument('--year', type=int, default=None, help='연도 (선택)')
    parser.add_argument('--output', type=str, default='artifacts/optuna', help='출력 디렉토리')
    
    args = parser.parse_args()
    
    candidates_path = Path(args.candidates)
    curated_path = Path(args.curated)
    output_path = Path(args.output)
    
    # 1. 후보 로드
    if not candidates_path.exists():
        print(f"❌ 튜닝 후보 파일을 찾을 수 없습니다: {candidates_path}")
        return
    
    candidates_df = pd.read_csv(candidates_path)
    print(f"📂 튜닝 후보 로드: {len(candidates_df)}개 시리즈")
    
    # 2. 튜닝 실행
    results_df = tune_series_parallel(
        candidates_df=candidates_df,
        curated_path=curated_path,
        timeout=args.timeout,
        n_trials=args.n_trials,
        max_workers=args.max_workers,
        max_series=args.max_series
    )
    
    # 3. 결과 저장
    print(f"\n{'='*70}")
    print("결과 저장 중...")
    print(f"{'='*70}\n")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 결과 CSV
    year_suffix = f"_{args.year}" if args.year else ""
    results_file = output_path / f'tuning_results{year_suffix}.csv'
    results_df.to_csv(results_file, index=False)
    print(f"✓ {results_file}")
    
    # 파라미터 JSON
    save_tuned_params(results_df, output_path, args.year)
    
    # 보고서
    generate_tuning_report(results_df, candidates_df, Path('reports'), args.year)
    
    # 4. 요약
    print(f"\n{'='*70}")
    print("튜닝 완료")
    print(f"{'='*70}\n")
    
    success_count = (results_df['status'] == 'success').sum()
    print(f"총 시리즈: {len(results_df)}")
    print(f"성공: {success_count}개")
    print(f"실패: {len(results_df) - success_count}개")
    
    if success_count > 0:
        avg_mase = results_df[results_df['status'] == 'success']['best_mase'].mean()
        print(f"\n평균 튜닝 후 MASE: {avg_mase:.4f}")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
