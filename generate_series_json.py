"""
Curated parquet → 시리즈별 JSON 파일 생성
증분학습 지원: lag_class, sample_weight 메타데이터 포함
"""
import pandas as pd
import json
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert curated parquet to series-level JSON files")
    parser.add_argument("--input", type=str, default="data/curated/claims_monthly.parquet",
                        help="Input curated parquet path")
    parser.add_argument("--output-dir", type=str, default="data/features",
                        help="Output directory for JSON files")
    parser.add_argument("--min-records", type=int, default=10,
                        help="Minimum non-zero records to create series file")
    args = parser.parse_args()
    
    print(f"[INFO] Loading curated data: {args.input}")
    df = pd.read_parquet(args.input)
    
    # 2021년 이후 데이터만 필터링 (학습 데이터 범위)
    print(f"[INFO] Original data: {len(df):,} rows, years {df['year'].min()}-{df['year'].max()}")
    df = df[df['year'] >= 2021].copy()
    print(f"[INFO] Filtered to 2021+: {len(df):,} rows, years {df['year'].min()}-{df['year'].max()}")
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Quality metadata 존재 여부 확인
    has_quality_meta = 'lag_class' in df.columns and 'sample_weight' in df.columns
    if has_quality_meta:
        print("[INFO] Quality metadata detected - will be included in JSON")
    
    # 시리즈별 그룹화
    series_groups = df.groupby('series_id')
    total_series = len(series_groups)
    created_count = 0
    skipped_count = 0
    
    print(f"[INFO] Processing {total_series} series...")
    
    for series_id, series_df in series_groups:
        # 최소 레코드 수 체크 (0이 아닌 클레임)
        non_zero_count = (series_df['claim_count'] > 0).sum()
        
        if non_zero_count < args.min_records:
            skipped_count += 1
            continue
        
        # 시리즈 정보 추출
        series_info = {
            "series_id": series_id,
            "plant": series_df['plant'].iloc[0],
            "product_cat2": series_df['product_cat2'].iloc[0],
            "mid_category": series_df['mid_category'].iloc[0],
            "records": len(series_df),
            "non_zero_records": int(non_zero_count),
            "total_claims": float(series_df['claim_count'].sum()),
            "mean_claims": float(series_df['claim_count'].mean()),
            "max_claims": float(series_df['claim_count'].max()),
        }
        
        # Quality metadata 통계
        if has_quality_meta:
            series_info["quality_stats"] = {
                "normal_ratio": float((series_df['lag_class'] == 'normal').sum() / len(series_df)),
                "borderline_ratio": float((series_df['lag_class'] == 'borderline').sum() / len(series_df)),
                "mean_sample_weight": float(series_df['sample_weight'].mean()),
            }
        
        # 시계열 데이터 (year, month, claim_count, 선택적으로 quality metadata)
        timeseries = series_df[['year', 'month', 'claim_count']].copy()
        
        if has_quality_meta:
            timeseries['lag_class'] = series_df['lag_class']
            timeseries['sample_weight'] = series_df['sample_weight']
        
        # 정렬 및 변환
        timeseries = timeseries.sort_values(['year', 'month'])
        series_info["data"] = timeseries.to_dict(orient='records')
        
        # JSON 파일 저장 (안전한 파일명 - Windows 금지 문자 제거)
        # Windows 금지 문자: < > : " / \ | ? *
        safe_filename = (series_id.replace('/', '_').replace('\\', '_').replace(':', '_')
                        .replace('|', '_').replace('?', '_').replace('*', '_')
                        .replace('<', '_').replace('>', '_').replace('"', '_'))
        json_path = output_dir / f"{safe_filename}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(series_info, f, ensure_ascii=False, indent=2)
        
        created_count += 1
        
        if created_count % 100 == 0:
            print(f"  ... {created_count}/{total_series} series processed")
    
    print(f"\n[SUCCESS] Created {created_count} JSON files in {output_dir}")
    print(f"[INFO] Skipped {skipped_count} series (< {args.min_records} non-zero records)")
    
    # 요약 통계 저장
    summary = {
        "total_series": total_series,
        "created_files": created_count,
        "skipped_series": skipped_count,
        "min_records_threshold": args.min_records,
        "has_quality_metadata": has_quality_meta,
        "input_file": args.input,
        "output_directory": str(output_dir),
    }
    
    summary_path = output_dir / "_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] Summary saved to {summary_path}")

if __name__ == '__main__':
    main()
