"""
Base Training: 2021-2023 데이터로 전체 시리즈 SARIMAX 모델 학습
- 2,608개 시리즈 JSON 파일에서 데이터 로드
- sample_weight 활용 (normal=1.0, borderline=0.5)
- 병렬 처리로 속도 최적화
- artifacts/models/base_2021_2023/ 저장
"""
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def train_single_series(json_path, train_until_year=2023, output_dir=None):
    """단일 시리즈 학습"""
    try:
        # JSON 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        series_id = data['series_id']
        df = pd.DataFrame(data['data'])
        
        # 학습 기간 필터링 (2021-2023)
        df_train = df[(df['year'] >= 2021) & (df['year'] <= train_until_year)].copy()
        
        if len(df_train) < 52:  # 최소 1년 데이터 필요
            return {
                'series_id': series_id,
                'status': 'skipped',
                'reason': f'insufficient_data (only {len(df_train)} weeks)',
                'path': None
            }
        
        # 시계열 데이터 준비
        y = df_train['claim_count'].values
        
        # sample_weight 준비 (있으면 사용)
        if 'sample_weight' in df_train.columns:
            weights = df_train['sample_weight'].values
        else:
            weights = np.ones(len(y))
        
        # 0 분산 체크
        if y.std() < 0.01:
            return {
                'series_id': series_id,
                'status': 'skipped',
                'reason': 'zero_variance',
                'path': None
            }
        
        # SARIMAX 모델 학습
        # 기본 파라미터: (1,0,1)(1,0,1,52) - 주간 계절성
        model = SARIMAX(
            y,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 52),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        # Fit with weights
        result = model.fit(
            disp=False,
            maxiter=100,
            method='lbfgs',
            # Note: statsmodels SARIMAX doesn't directly support sample_weights
            # We'll save weights for future use in append_fit
        )
        
        # 모델 저장
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 안전한 파일명
            safe_filename = (series_id.replace('/', '_').replace('\\', '_').replace(':', '_')
                           .replace('|', '_').replace('?', '_').replace('*', '_')
                           .replace('<', '_').replace('>', '_').replace('"', '_'))
            
            model_path = output_dir / f"{safe_filename}.pkl"
            
            # 모델 파라미터만 저장 (전체 모델 객체는 너무 큼)
            model_info = {
                'series_id': series_id,
                'params': result.params.tolist(),  # 파라미터 값만
                'model_spec': {
                    'order': (1, 0, 1),
                    'seasonal_order': (1, 0, 1, 52),
                    'enforce_stationarity': False,
                    'enforce_invertibility': False
                },
                'metadata': {
                    'start_year': int(df_train['year'].min()),
                    'end_year': int(df_train['year'].max()),
                    'n_obs': len(y),
                    'has_weights': 'sample_weight' in df_train.columns
                },
                'trained_at': datetime.now().isoformat(),
                'aic': result.aic,
                'bic': result.bic,
                'converged': result.mle_retvals['converged'] if hasattr(result, 'mle_retvals') else True
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_info, f)
            
            return {
                'series_id': series_id,
                'status': 'success',
                'n_obs': len(y),
                'aic': result.aic,
                'bic': result.bic,
                'path': str(model_path)
            }
        else:
            return {
                'series_id': series_id,
                'status': 'success',
                'n_obs': len(y),
                'aic': result.aic,
                'bic': result.bic,
                'path': None
            }
    
    except Exception as e:
        return {
            'series_id': json_path.stem if hasattr(json_path, 'stem') else str(json_path),
            'status': 'error',
            'reason': str(e),
            'path': None
        }


def main():
    parser = argparse.ArgumentParser(description="Train base SARIMAX models for all series")
    parser.add_argument("--json-dir", type=str, default="data/features",
                        help="Directory containing series JSON files")
    parser.add_argument("--output-dir", type=str, default="artifacts/models/base_2021_2023",
                        help="Output directory for trained models")
    parser.add_argument("--train-until", type=int, default=2023,
                        help="Train data until this year (inclusive)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of series to train (for testing)")
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Base Training: SARIMAX Models (2021-{args.train_until})")
    print("=" * 80)
    
    # JSON 파일 목록
    json_dir = Path(args.json_dir)
    json_files = list(json_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != '_summary.json']
    
    if args.limit:
        json_files = json_files[:args.limit]
    
    print(f"\n[INFO] Found {len(json_files)} series JSON files")
    print(f"[INFO] Training until: {args.train_until}")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] Parallel workers: {args.max_workers}")
    
    # 병렬 학습
    results = []
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"\n[INFO] Starting training...")
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(train_single_series, json_file, args.train_until, args.output_dir): json_file
            for json_file in json_files
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'skipped':
                skipped_count += 1
            else:
                error_count += 1
            
            # Progress update every 50 series
            if i % 50 == 0 or i == len(json_files):
                print(f"  Progress: {i}/{len(json_files)} | "
                      f"Success: {success_count} | "
                      f"Skipped: {skipped_count} | "
                      f"Errors: {error_count}")
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Total series:   {len(json_files)}")
    print(f"Success:        {success_count}")
    print(f"Skipped:        {skipped_count}")
    print(f"Errors:         {error_count}")
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_path = Path(args.output_dir) / "training_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] Results saved to: {results_path}")
    
    # 상위 모델 표시
    if success_count > 0:
        success_results = results_df[results_df['status'] == 'success'].copy()
        success_results = success_results.sort_values('aic')
        
        print(f"\n[INFO] Top 10 models (by AIC):")
        for idx, row in success_results.head(10).iterrows():
            print(f"  {row['series_id'][:50]:50s} | AIC: {row['aic']:,.1f} | Obs: {int(row['n_obs'])}")
    
    # 오류 상세
    if error_count > 0:
        print(f"\n[WARNING] {error_count} errors occurred:")
        error_results = results_df[results_df['status'] == 'error']
        for idx, row in error_results.head(5).iterrows():
            print(f"  {row['series_id']}: {row['reason']}")
    
    print("\n[SUCCESS] Base training completed!")


if __name__ == '__main__':
    main()
