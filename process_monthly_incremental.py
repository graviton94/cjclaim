# -*- coding: utf-8 -*-
"""
Incremental monthly update process
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.preprocess import monthly_agg_from_counts

def append_new_month(master_df, csv_path, year, month):
    df_new = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"  Raw records: {len(df_new):,}")
    
    # 컬럼명 매핑 (업로드 CSV → 내부 형식)
    column_mapping = {
        '발생일자': 'occurrence_date',  # 발생일자는 사용 안 하지만 매핑
        '중분류': 'mid_category',  # 중분류
        '중분류(보정)': 'mid_category',  # 중분류(보정) - 둘 다 지원
        '플랜트': 'plant',
        '제품범주2': 'product_cat2',
        '제조일자': 'manufacturing_date',
        'count': 'claim_count'
    }
    
    # 실제 존재하는 컬럼만 rename
    rename_dict = {k: v for k, v in column_mapping.items() if k in df_new.columns}
    df_new = df_new.rename(columns=rename_dict)
    
    # 디버깅: 매핑 후 샘플 데이터 출력
    print(f"  Columns after mapping: {df_new.columns.tolist()}")
    print(f"  Sample data (first 3 rows):")
    print(f"    {df_new[['plant', 'product_cat2', 'mid_category', 'manufacturing_date']].head(3).to_dict('records')}")
    
    # 제조일자가 이미 YYYY-MM-DD 형식이므로 그대로 사용
    # monthly_agg_from_counts에 명시적으로 date_col 전달
    
    # 영어 컬럼명으로 group_cols 명시
    df_monthly = monthly_agg_from_counts(
        df_new, 
        date_col='manufacturing_date',  # 매핑된 컬럼명 사용
        value_col='claim_count',
        group_cols=['plant', 'product_cat2', 'mid_category'],  # 영어 컬럼명
        min_claims=0, 
        pad_to_date=(year, month)
    )
    
    print(f"  Aggregated series: {df_monthly['series_id'].nunique()}")
    print(f"  Aggregated rows: {len(df_monthly)}")
    df_monthly = df_monthly[(df_monthly['year'] == year) & (df_monthly['month'] == month)].copy()
    
    if not master_df.empty:
        master_df = master_df[~((master_df['year'] == year) & (master_df['month'] == month))].copy()
        updated_df = pd.concat([master_df, df_monthly], ignore_index=True)
    else:
        updated_df = df_monthly
    
    return updated_df.sort_values(['series_id', 'year', 'month']).reset_index(drop=True)

def update_json(series_id, series_df, json_dir):
    safe_name = series_id.replace('/', '_').replace('\\', '_').replace(':', '_').replace('|', '_')
    json_path = json_dir / f"{safe_name}.json"
    
    info = {
        "series_id": series_id,
        "plant": series_df['plant'].iloc[0],
        "product_cat2": series_df['product_cat2'].iloc[0],
        "mid_category": series_df['mid_category'].iloc[0],
        "total_claims": float(series_df['claim_count'].sum()),
        "updated_at": datetime.now().isoformat(),
        "data": series_df[['year', 'month', 'claim_count']].sort_values(['year', 'month']).to_dict(orient='records')
    }
    
    action = 'updated' if json_path.exists() else 'created'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return action

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-csv", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--master-parquet", default="data/curated/claims_monthly.parquet")
    parser.add_argument("--json-dir", default="data/features")
    parser.add_argument("--min-claims", type=int, default=10)
    parser.add_argument("--output-list", default=None)
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Incremental Update: {args.year}-{args.month:02d}")
    print("=" * 80)
    
    # Load master
    master_path = Path(args.master_parquet)
    if master_path.exists():
        master_df = pd.read_parquet(master_path)
        print(f"[INFO] Master loaded: {len(master_df)} rows")
    else:
        master_df = pd.DataFrame()
    
    # Append new month
    print(f"\n[INFO] Processing {args.new_csv}")
    updated_df = append_new_month(master_df, args.new_csv, args.year, args.month)
    
    # Save
    updated_df.to_parquet(master_path, index=False)
    print(f"[SUCCESS] Updated: {len(updated_df)} rows, {updated_df['series_id'].nunique()} series")
    
    # Filter viable series
    totals = updated_df.groupby('series_id')['claim_count'].sum().reset_index()
    totals.columns = ['series_id', 'total_claims']
    viable = totals[totals['total_claims'] >= args.min_claims]
    print(f"\n[INFO] Viable series (>={args.min_claims}): {len(viable)}/{len(totals)}")
    
    # Update JSON
    json_dir = Path(args.json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    
    created, updated_list = 0, []
    for idx, row in viable.iterrows():
        series_id = row['series_id']
        series_df = updated_df[updated_df['series_id'] == series_id]
        action = update_json(series_id, series_df, json_dir)
        if action == 'created':
            created += 1
            print(f"  [NEW] {series_id}")
        updated_list.append(series_id)
    
    print(f"\n[SUCCESS] JSON updates: {created} created, {len(updated_list)} total")
    
    # Save list
    if args.output_list:
        Path(args.output_list).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_list).write_text('\n'.join(updated_list), encoding='utf-8')
        print(f"[INFO] List saved: {args.output_list}")
    
    # 파이프라인 결과 저장 경로
    incremental_dir = Path(f"artifacts/incremental/{args.year}{args.month:02d}")
    incremental_dir.mkdir(parents=True, exist_ok=True)

    # 1. 예측 결과 생성
    forecast_path = incremental_dir / "forecast.parquet"
    import subprocess
    subprocess.run([
        sys.executable, "pipeline_forecast.py",
        "--year", str(args.year),
        "--month", str(args.month),
        "--output", str(forecast_path)
    ], check=True)

    # 2. 평가 결과 생성
    evaluation_path = incremental_dir / "evaluation.json"
    subprocess.run([
        sys.executable, "evaluate_predictions.py",
        "--year", str(args.year),
        "--month", str(args.month),
        "--output", str(evaluation_path)
    ], check=True)

    # 3. 모델 재학습 결과 생성
    model_path = incremental_dir / "model.pkl"
    subprocess.run([
        sys.executable, "train_incremental_models.py",
        "--year", str(args.year),
        "--month", str(args.month),
        "--output", str(model_path)
    ], check=True)

    # 4. 결과 파일 생성 여부 검증
    required_files = [forecast_path, evaluation_path, model_path]
    missing = [str(f) for f in required_files if not f.exists()]
    if missing:
        print(f"[ERROR] Missing files: {missing}")
    else:
        print("[SUCCESS] All result files exist.")
    return 0

if __name__ == '__main__':
    exit(main())
