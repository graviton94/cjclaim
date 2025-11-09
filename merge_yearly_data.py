"""
연도별 분할 Raw Data 병합 스크립트
====================================
C:\cjclaim\data 에서 년도별 CSV 파일들을 읽어서
quality-cycles\data\raw 에 통합 파일로 저장

사용법:
    python merge_yearly_data.py
    python merge_yearly_data.py --source C:\cjclaim\data --years 2021 2022 2023 2024
    python merge_yearly_data.py --pattern "claims_*.csv"
"""
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def detect_encoding(file_path):
    """CSV 파일 인코딩 자동 탐지"""
    encodings = ['utf-8-sig', 'utf-8', 'euc-kr', 'cp949']
    
    for enc in encodings:
        try:
            pd.read_csv(file_path, encoding=enc, nrows=5)
            return enc
        except:
            continue
    
    return 'utf-8'  # fallback


def load_yearly_file(file_path, encoding=None):
    """단일 연도 파일 로드"""
    if encoding is None:
        encoding = detect_encoding(file_path)
    
    print(f"  Loading {file_path.name} (encoding: {encoding})...")
    
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"    → {len(df):,} rows loaded")
        return df, encoding
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None, encoding


def merge_yearly_data(
    source_dir: str = "C:/cjclaim/data",
    output_path: str = "data/raw/claims_merged.csv",
    years: list = None,
    file_pattern: str = None,
    deduplicate: bool = True
):
    """
    여러 연도별 파일을 병합
    
    Args:
        source_dir: 원본 데이터 디렉토리
        output_path: 출력 파일 경로
        years: 병합할 연도 리스트 (None이면 자동 탐지)
        file_pattern: 파일명 패턴 (e.g., "claims_*.csv")
        deduplicate: 중복 제거 여부
    
    Returns:
        병합된 DataFrame
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    print("=" * 80)
    print("Yearly Data Merge Pipeline")
    print("=" * 80)
    print(f"Source directory: {source_path}")
    print(f"Output path: {output_path}")
    print()
    
    # 파일 목록 수집
    if file_pattern:
        # 패턴 기반
        files = sorted(source_path.glob(file_pattern))
    elif years:
        # 연도 기반
        files = []
        for year in years:
            # 여러 가능한 파일명 패턴 시도
            patterns = [
                f"{year}_raw.csv",
                f"claims_{year}.csv",
                f"{year}.csv",
                f"data_{year}.csv"
            ]
            for pattern in patterns:
                matches = list(source_path.glob(pattern))
                if matches:
                    files.extend(matches)
                    break
    else:
        # 모든 CSV 파일
        files = sorted(source_path.glob("*.csv"))
    
    if not files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")
    
    print(f"Found {len(files)} file(s) to merge:")
    for f in files:
        print(f"  - {f.name}")
    print()
    
    # 순차적으로 로드 및 병합
    all_dfs = []
    total_rows = 0
    encoding_used = None
    
    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}]", end=" ")
        df, enc = load_yearly_file(file_path, encoding_used)
        
        if df is not None:
            all_dfs.append(df)
            total_rows += len(df)
            encoding_used = enc  # 같은 인코딩 재사용
    
    if not all_dfs:
        raise ValueError("No data loaded successfully")
    
    # 병합
    print(f"\n{'='*80}")
    print(f"Merging {len(all_dfs)} DataFrames...")
    df_merged = pd.concat(all_dfs, ignore_index=True)
    
    print(f"  Total rows before processing: {len(df_merged):,}")
    
    # 중복 처리: drop_duplicates 대신 groupby로 count 합산
    if deduplicate:
        original_len = len(df_merged)
        
        # 그룹화 키 결정 (발생일자, 중분류, 플랜트, 제품범주2, 제조일자)
        group_cols = []
        for col in ['발생일자', '중분류', '플랜트', '제품범주2', '제조일자']:
            if col in df_merged.columns:
                group_cols.append(col)
        
        # count 컬럼 확인
        count_col = None
        for col in ['count', 'claim_count', 'y']:
            if col in df_merged.columns:
                count_col = col
                break
        
        if len(group_cols) >= 4 and count_col:
            # 같은 키에 대해 count 값을 합산 (중복이 아니라 실제 발생 건수)
            df_merged = df_merged.groupby(group_cols, as_index=False)[count_col].sum()
            new_len = len(df_merged)
            print(f"  Aggregated {original_len:,} rows → {new_len:,} unique series-date combinations")
            print(f"  (Same series+date occurrences were summed, not removed)")
        else:
            print(f"  ⚠️  Insufficient columns for aggregation, keeping all rows")
    
    print(f"  Final rows: {len(df_merged):,}")
    
    # 데이터 품질 체크
    print(f"\n{'='*80}")
    print("Data Quality Check")
    print(f"{'='*80}")
    print(f"Columns: {len(df_merged.columns)}")
    print(f"  {', '.join(df_merged.columns[:10].tolist())}...")
    
    # 날짜 컬럼 확인
    date_col = None
    for col in ['제조일자', 'date', '발생일', 'occurrence_date']:
        if col in df_merged.columns:
            date_col = col
            break
    
    if date_col:
        # 날짜 파싱 시도
        try:
            df_merged[date_col] = pd.to_datetime(df_merged[date_col], errors='coerce')
            df_merged['year'] = df_merged[date_col].dt.year
            df_merged['month'] = df_merged[date_col].dt.month
            
            year_counts = df_merged['year'].value_counts().sort_index()
            print(f"\nYear distribution:")
            for year, count in year_counts.items():
                if pd.notna(year):
                    print(f"  {int(year)}: {count:,} rows")
        except:
            print(f"  Warning: Could not parse date column '{date_col}'")
    
    # 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Saving to: {output_path}")
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Rows: {len(df_merged):,}")
    print(f"  Columns: {len(df_merged.columns)}")
    
    print(f"\n✅ SUCCESS: Merged data saved!")
    print(f"{'='*80}")
    
    return df_merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge yearly CSV files into single dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect all CSV files in C:\cjclaim\data
  python merge_yearly_data.py
  
  # Specify years explicitly
  python merge_yearly_data.py --years 2021 2022 2023 2024
  
  # Use file pattern
  python merge_yearly_data.py --pattern "claims_*.csv"
  
  # Custom source directory
  python merge_yearly_data.py --source D:\backup\claims_data
  
  # No deduplication
  python merge_yearly_data.py --no-deduplicate
        """
    )
    
    parser.add_argument("--source", type=str, default="C:/cjclaim/data",
                        help="Source directory containing yearly CSV files (default: C:/cjclaim/data)")
    parser.add_argument("--output", type=str, default="data/raw/claims_merged.csv",
                        help="Output merged CSV path (default: data/raw/claims_merged.csv)")
    parser.add_argument("--years", type=int, nargs='+',
                        help="Specific years to merge (e.g., --years 2021 2022 2023)")
    parser.add_argument("--pattern", type=str,
                        help="File name pattern (e.g., --pattern 'claims_*.csv')")
    parser.add_argument("--no-deduplicate", action="store_true",
                        help="Skip duplicate removal")
    
    args = parser.parse_args()
    
    try:
        df_merged = merge_yearly_data(
            source_dir=args.source,
            output_path=args.output,
            years=args.years,
            file_pattern=args.pattern,
            deduplicate=not args.no_deduplicate
        )
        
        # 다음 단계 안내
        print(f"\n📋 Next Steps:")
        print(f"1. Verify data: head data/raw/claims_merged.csv")
        print(f"2. Preprocess: python preprocess_to_curated.py --input {args.output}")
        print(f"3. Generate JSONs: python generate_series_json.py")
        print(f"4. Train models: python train_base_models.py --auto-optimize")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
