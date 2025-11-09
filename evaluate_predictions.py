"""
예측값 vs 실제값 비교 평가
- 2025년 데이터 업로드 시 기존 예측과 비교
- 각 시리즈별 예측 성능 계산 및 저장
"""
import pandas as pd
import json
from pathlib import Path
import numpy as np
from datetime import datetime


def calculate_metrics(actual, predicted):
    """예측 성능 지표 계산"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # 0으로 나누는 것 방지
    mask = actual != 0
    
    # MAPE (Mean Absolute Percentage Error)
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = None
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(actual - predicted))
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # R² Score
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else None
    
    return {
        'mape': float(mape) if mape is not None else None,
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2) if r2 is not None else None,
        'n_points': int(len(actual))
    }


def evaluate_predictions(year, output_path=None):
    """
    기존 예측값과 새로 추가된 실제값을 비교하여 성능 평가
    """
    print("=" * 80)
    print("예측 성능 평가 시작")
    print("=" * 80)
    
    # 경로 설정
    curated_path = Path('data/curated/claims.parquet')
    models_dir = Path('artifacts/models')
    eval_dir = Path('artifacts/evaluations')
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    if not curated_path.exists():
        print("❌ Curated 데이터가 없습니다.")
        return
    
    if not models_dir.exists():
        print("❌ 학습된 모델이 없습니다.")
        return
    
    # Curated 데이터 로드
    df = pd.read_parquet(curated_path)
    
    # 해당 연도 데이터만 추출 (평가 대상)
    df_year = df[df['year'] == year].copy()
    
    if len(df_year) == 0:
        print(f"ℹ️ {year}년 데이터가 없습니다. 평가를 건너뜁니다.")
        return
    
    print(f"📊 {year}년 데이터: {len(df_year):,}행")
    
    # 주차 범위 확인
    weeks_year = sorted(df_year['week'].unique())
    print(f"📅 {year}년 주차 범위: W{min(weeks_year):02d} ~ W{max(weeks_year):02d} ({len(weeks_year)}주)")
    
    # 각 시리즈별 평가
    evaluations = []
    series_list = df_year['series_id'].unique()
    
    print(f"\n🔍 {len(series_list)}개 시리즈 평가 중...")
    
    for i, series_id in enumerate(series_list, 1):
        if i % 100 == 0:
            print(f"  진행: {i}/{len(series_list)}")
        
        # 안전한 파일명 생성
        safe_filename = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in series_id)
        model_path = models_dir / f"{safe_filename}.json"
        
        if not model_path.exists():
            continue
        
        # 모델 데이터 로드
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        except:
            continue
        
        # 예측값 추출
        if 'forecast' not in model_data:
            continue
        
        forecast = model_data['forecast']
        predicted_values = forecast['yhat']  # 26주 예측값
        
        # 실제값 추출 (해당 연도 해당 시리즈)
        df_series = df_year[df_year['series_id'] == series_id].copy()
        df_series = df_series.sort_values('week')
        
        actual_weeks = df_series['week'].tolist()
        actual_values = df_series['claim_count'].tolist()
        
        # 예측값과 실제값 매칭 (주차 기준)
        # 예측은 2025-W01부터 시작한다고 가정
        matched_predicted = []
        matched_actual = []
        
        for week, actual_count in zip(actual_weeks, actual_values):
            # week는 1부터 시작 (2025-W01 = 1)
            pred_idx = week - 1  # 0-based index
            
            if 0 <= pred_idx < len(predicted_values):
                matched_predicted.append(predicted_values[pred_idx])
                matched_actual.append(actual_count)
        
        if len(matched_actual) == 0:
            continue
        
        # 성능 지표 계산
        metrics = calculate_metrics(matched_actual, matched_predicted)
        
        evaluations.append({
            'series_id': series_id,
            'plant': series_id.split('|')[0] if '|' in series_id else 'Unknown',
            'product_cat2': series_id.split('|')[1] if '|' in series_id and len(series_id.split('|')) > 1 else 'Unknown',
            'mid_category': series_id.split('|')[2] if '|' in series_id and len(series_id.split('|')) > 2 else 'Unknown',
            'weeks_evaluated': actual_weeks,
            'actual_values': matched_actual,
            'predicted_values': matched_predicted,
            'metrics': metrics,
            'evaluation_date': datetime.now().isoformat()
        })
    
    print(f"\n✅ {len(evaluations)}개 시리즈 평가 완료")
    
    if len(evaluations) == 0:
        print("⚠️ 평가 결과가 없습니다.")
        return
    
    # 결과 저장
    result = {
        'evaluation_date': datetime.now().isoformat(),
        'weeks_range': {'min': int(min(weeks_year)), 'max': int(max(weeks_year))},
        'n_series': len(evaluations),
        'evaluations': evaluations
    }
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        eval_path = eval_dir / f"evaluation_{year}.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 평가 결과 저장: {eval_path}")
    
    # 요약 통계
    all_mapes = [e['metrics']['mape'] for e in evaluations if e['metrics']['mape'] is not None]
    all_maes = [e['metrics']['mae'] for e in evaluations]
    all_rmses = [e['metrics']['rmse'] for e in evaluations]
    
    print("\n" + "=" * 80)
    print("📈 평가 요약")
    print("=" * 80)
    print(f"평가 시리즈: {len(evaluations)}개")
    print(f"평가 주차: W{min(weeks_year):02d} ~ W{max(weeks_year):02d}")
    
    if all_mapes:
        print(f"\nMAPE 평균: {np.mean(all_mapes):.2f}%")
        print(f"MAPE 중앙값: {np.median(all_mapes):.2f}%")
    
    print(f"\nMAE 평균: {np.mean(all_maes):.4f}")
    print(f"MAE 중앙값: {np.median(all_maes):.4f}")
    
    print(f"\nRMSE 평균: {np.mean(all_rmses):.4f}")
    print(f"RMSE 중앙값: {np.median(all_rmses):.4f}")
    
    # Top/Bottom 시리즈
    if all_mapes:
        sorted_by_mape = sorted(evaluations, key=lambda x: x['metrics']['mape'] if x['metrics']['mape'] is not None else float('inf'))
        
        print("\n🏆 예측 정확도 Top 5:")
        for i, e in enumerate(sorted_by_mape[:5], 1):
            print(f"  {i}. {e['series_id']}: MAPE {e['metrics']['mape']:.2f}%")
        
        print("\n⚠️ 예측 정확도 Bottom 5:")
        for i, e in enumerate(sorted_by_mape[-5:][::-1], 1):
            mape = e['metrics']['mape'] if e['metrics']['mape'] is not None else 'N/A'
            print(f"  {i}. {e['series_id']}: MAPE {mape if mape == 'N/A' else f'{mape:.2f}%'}")
    
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=False)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    evaluate_predictions(args.year, args.output)
