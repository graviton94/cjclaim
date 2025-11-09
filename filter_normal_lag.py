"""
Lag 정상범주 분석 및 이상치 필터링
=====================================
제품범주별로 (발생일자 - 제조일자) 분포를 분석하여
정상 범주(IQR 또는 percentile 기반) 내 데이터만 학습에 사용

주요 기능:
1. 제품범주2별 lag 분포 통계 계산
2. 이상치 탐지 (IQR method 또는 percentile)
3. 정상 범주 내 데이터 필터링
4. 필터링 리포트 생성

사용법:
    python filter_normal_lag.py --input data/raw/claims_merged.csv
    python filter_normal_lag.py --method percentile --lower 5 --upper 95
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def calculate_lag_days(df, date_col, mfg_col):
    """
    발생일자 - 제조일자 = lag (일 단위)
    
    Args:
        df: DataFrame
        date_col: 발생일자 컬럼명
        mfg_col: 제조일자 컬럼명
    
    Returns:
        DataFrame with 'lag_days' column
    """
    df = df.copy()
    
    # 날짜 파싱
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[mfg_col] = pd.to_datetime(df[mfg_col], errors='coerce')
    
    # Lag 계산
    df['lag_days'] = (df[date_col] - df[mfg_col]).dt.days
    
    return df


def analyze_lag_distribution(df, product_col='제품범주2'):
    """
    제품범주별 lag 분포 통계 분석
    
    Returns:
        DataFrame with columns: product, count, mean, std, min, q25, median, q75, max, iqr
    """
    stats = []
    
    for product in df[product_col].unique():
        if pd.isna(product):
            continue
        
        subset = df[df[product_col] == product]['lag_days'].dropna()
        
        if len(subset) < 10:  # 최소 10개 데이터 필요
            continue
        
        q25, q50, q75 = subset.quantile([0.25, 0.5, 0.75])
        iqr = q75 - q25
        
        stats.append({
            'product': product,
            'count': len(subset),
            'mean': subset.mean(),
            'std': subset.std(),
            'min': subset.min(),
            'q25': q25,
            'median': q50,
            'q75': q75,
            'max': subset.max(),
            'iqr': iqr,
            'lower_fence': q25 - 1.5 * iqr,  # IQR method
            'upper_fence': q75 + 1.5 * iqr
        })
    
    return pd.DataFrame(stats).sort_values('count', ascending=False)


def filter_normal_lag_iqr(df, product_col='제품범주2', multiplier=1.5):
    """
    IQR 방식으로 정상 범주 필터링
    
    정상 범주: [Q1 - multiplier*IQR, Q3 + multiplier*IQR]
    
    Args:
        df: DataFrame with 'lag_days' column
        product_col: 제품범주 컬럼명
        multiplier: IQR 승수 (1.5=moderate, 3.0=extreme outliers)
    
    Returns:
        Filtered DataFrame, filter stats
    """
    df = df.copy()
    df['is_normal_lag'] = False
    
    filter_stats = []
    
    for product in df[product_col].unique():
        if pd.isna(product):
            continue
        
        mask = (df[product_col] == product) & df['lag_days'].notna()
        subset = df.loc[mask, 'lag_days']
        
        if len(subset) < 10:
            # 데이터 부족 → 모두 정상으로 간주
            df.loc[mask, 'is_normal_lag'] = True
            continue
        
        q25, q75 = subset.quantile([0.25, 0.75])
        iqr = q75 - q25
        
        lower_fence = q25 - multiplier * iqr
        upper_fence = q75 + multiplier * iqr
        
        # 정상 범주 내 데이터 마킹
        normal_mask = mask & (df['lag_days'] >= lower_fence) & (df['lag_days'] <= upper_fence)
        df.loc[normal_mask, 'is_normal_lag'] = True
        
        # 통계 기록
        total = mask.sum()
        normal = normal_mask.sum()
        outliers = total - normal
        
        filter_stats.append({
            'product': product,
            'total': total,
            'normal': normal,
            'outliers': outliers,
            'outlier_ratio': outliers / total if total > 0 else 0,
            'lower_fence': lower_fence,
            'upper_fence': upper_fence,
            'q25': q25,
            'q75': q75,
            'iqr': iqr
        })
    
    df_filtered = df[df['is_normal_lag']].copy()
    
    return df_filtered, pd.DataFrame(filter_stats)


def filter_normal_lag_percentile(df, product_col='제품범주2', lower=5, upper=95):
    """
    Percentile 방식으로 정상 범주 필터링
    
    정상 범주: [lower_percentile, upper_percentile]
    
    Args:
        df: DataFrame with 'lag_days' column
        product_col: 제품범주 컬럼명
        lower: 하위 percentile (default: 5%)
        upper: 상위 percentile (default: 95%)
    
    Returns:
        Filtered DataFrame, filter stats
    """
    df = df.copy()
    df['is_normal_lag'] = False
    
    filter_stats = []
    
    for product in df[product_col].unique():
        if pd.isna(product):
            continue
        
        mask = (df[product_col] == product) & df['lag_days'].notna()
        subset = df.loc[mask, 'lag_days']
        
        if len(subset) < 10:
            df.loc[mask, 'is_normal_lag'] = True
            continue
        
        lower_bound = subset.quantile(lower / 100)
        upper_bound = subset.quantile(upper / 100)
        
        # 정상 범주 내 데이터 마킹
        normal_mask = mask & (df['lag_days'] >= lower_bound) & (df['lag_days'] <= upper_bound)
        df.loc[normal_mask, 'is_normal_lag'] = True
        
        # 통계 기록
        total = mask.sum()
        normal = normal_mask.sum()
        outliers = total - normal
        
        filter_stats.append({
            'product': product,
            'total': total,
            'normal': normal,
            'outliers': outliers,
            'outlier_ratio': outliers / total if total > 0 else 0,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'p' + str(lower): lower_bound,
            'p' + str(upper): upper_bound
        })
    
    df_filtered = df[df['is_normal_lag']].copy()
    
    return df_filtered, pd.DataFrame(filter_stats)


def generate_filter_report(df_original, df_filtered, stats_df, output_path):
    """필터링 리포트 생성"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'original_rows': len(df_original),
            'filtered_rows': len(df_filtered),
            'removed_rows': len(df_original) - len(df_filtered),
            'removal_ratio': (len(df_original) - len(df_filtered)) / len(df_original),
            'products_analyzed': len(stats_df)
        },
        'top_outlier_products': stats_df.nlargest(10, 'outlier_ratio')[
            ['product', 'total', 'outliers', 'outlier_ratio']
        ].to_dict('records'),
        'overall_stats': {
            'mean_outlier_ratio': stats_df['outlier_ratio'].mean(),
            'median_outlier_ratio': stats_df['outlier_ratio'].median(),
            'max_outlier_ratio': stats_df['outlier_ratio'].max()
        }
    }
    
    # Save JSON report
    report_path = Path(output_path).parent / 'lag_filter_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save detailed stats CSV
    stats_path = Path(output_path).parent / 'lag_filter_stats.csv'
    stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
    
    return report, report_path, stats_path


def main():
    parser = argparse.ArgumentParser(
        description="Filter claims data by normal lag range per product category",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # IQR method (default: 1.5x IQR)
  python filter_normal_lag.py --input data/raw/claims_merged.csv
  
  # Stricter IQR (3.0x for extreme outliers only)
  python filter_normal_lag.py --method iqr --multiplier 3.0
  
  # Percentile method (keep 5th-95th percentile)
  python filter_normal_lag.py --method percentile --lower 5 --upper 95
  
  # Custom output path
  python filter_normal_lag.py --output data/raw/claims_normal_lag.csv
        """
    )
    
    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV file path")
    parser.add_argument("--output", type=str, default="data/raw/claims_normal_lag.csv",
                        help="Output filtered CSV path")
    parser.add_argument("--method", choices=['iqr', 'percentile'], default='iqr',
                        help="Outlier detection method (default: iqr)")
    parser.add_argument("--multiplier", type=float, default=1.5,
                        help="IQR multiplier for outlier detection (default: 1.5)")
    parser.add_argument("--lower", type=float, default=5,
                        help="Lower percentile for percentile method (default: 5)")
    parser.add_argument("--upper", type=float, default=95,
                        help="Upper percentile for percentile method (default: 95)")
    parser.add_argument("--date-col", type=str, default=None,
                        help="발생일자 column name (auto-detect if not specified)")
    parser.add_argument("--mfg-col", type=str, default=None,
                        help="제조일자 column name (auto-detect if not specified)")
    parser.add_argument("--product-col", type=str, default=None,
                        help="제품범주 column name (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Lag 정상범주 필터링 파이프라인")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Method: {args.method.upper()}")
    if args.method == 'iqr':
        print(f"IQR Multiplier: {args.multiplier}")
    else:
        print(f"Percentile Range: {args.lower}th - {args.upper}th")
    print("=" * 80)
    print()
    
    # Load data
    print("[1/5] Loading data...")
    for enc in ['utf-8-sig', 'utf-8', 'euc-kr', 'cp949']:
        try:
            df = pd.read_csv(args.input, encoding=enc)
            print(f"  ✓ Loaded {len(df):,} rows (encoding: {enc})")
            break
        except:
            continue
    
    # Auto-detect columns
    print("\n[2/5] Detecting columns...")
    
    # 발생일자
    if args.date_col:
        date_col = args.date_col
    else:
        date_candidates = ['발생일자', 'date', '클레임일자', 'claim_date']
        date_col = next((c for c in date_candidates if c in df.columns), None)
    
    # 제조일자
    if args.mfg_col:
        mfg_col = args.mfg_col
    else:
        mfg_candidates = ['제조일자', 'mfg_date', '생산일자', 'production_date']
        mfg_col = next((c for c in mfg_candidates if c in df.columns), None)
    
    # 제품범주
    if args.product_col:
        product_col = args.product_col
    else:
        product_candidates = ['제품범주2', 'product_cat2', '제품범주', 'product']
        product_col = next((c for c in product_candidates if c in df.columns), None)
    
    if not date_col or not mfg_col:
        raise ValueError(f"Could not find date columns. Found: {df.columns.tolist()}")
    
    if not product_col:
        raise ValueError(f"Could not find product column. Found: {df.columns.tolist()}")
    
    print(f"  발생일자: {date_col}")
    print(f"  제조일자: {mfg_col}")
    print(f"  제품범주: {product_col}")
    
    # Calculate lag
    print("\n[3/5] Calculating lag days...")
    df = calculate_lag_days(df, date_col, mfg_col)
    
    valid_lag = df['lag_days'].notna().sum()
    print(f"  Valid lag calculations: {valid_lag:,} / {len(df):,} ({valid_lag/len(df)*100:.1f}%)")
    
    # Analyze distribution
    print("\n[4/5] Analyzing lag distribution...")
    stats_df = analyze_lag_distribution(df, product_col)
    print(f"  Analyzed {len(stats_df)} product categories")
    print(f"\n  Top 5 products by volume:")
    for _, row in stats_df.head(5).iterrows():
        print(f"    {row['product'][:40]:40s} | Count: {int(row['count']):>6,} | "
              f"Median: {row['median']:>6.0f} days | IQR: {row['iqr']:>6.0f}")
    
    # Filter outliers
    print(f"\n[5/5] Filtering outliers ({args.method} method)...")
    
    if args.method == 'iqr':
        df_filtered, filter_stats = filter_normal_lag_iqr(
            df, product_col, multiplier=args.multiplier
        )
    else:
        df_filtered, filter_stats = filter_normal_lag_percentile(
            df, product_col, lower=args.lower, upper=args.upper
        )
    
    # Summary
    original_count = len(df)
    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count
    
    print(f"\n{'='*80}")
    print("Filtering Results")
    print(f"{'='*80}")
    print(f"Original rows:  {original_count:,}")
    print(f"Filtered rows:  {filtered_count:,}")
    print(f"Removed rows:   {removed_count:,} ({removed_count/original_count*100:.2f}%)")
    print()
    
    # Top outlier products
    top_outliers = filter_stats.nlargest(5, 'outlier_ratio')
    print("Top 5 products by outlier ratio:")
    for _, row in top_outliers.iterrows():
        print(f"  {row['product'][:40]:40s} | "
              f"Total: {int(row['total']):>5,} | "
              f"Outliers: {int(row['outliers']):>4,} ({row['outlier_ratio']*100:>5.1f}%)")
    
    # Save filtered data
    print(f"\n{'='*80}")
    print("Saving outputs...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove temporary columns before saving
    cols_to_drop = ['lag_days', 'is_normal_lag']
    df_filtered = df_filtered.drop(columns=[c for c in cols_to_drop if c in df_filtered.columns])
    
    df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ Filtered data: {output_path}")
    
    # Generate report
    report, report_path, stats_path = generate_filter_report(
        df, df_filtered, filter_stats, output_path
    )
    print(f"  ✓ Filter report: {report_path}")
    print(f"  ✓ Filter stats:  {stats_path}")
    
    print(f"\n{'='*80}")
    print("✅ Lag filtering completed successfully!")
    print(f"{'='*80}")
    print("\n📋 Next Steps:")
    print(f"1. Review report: {report_path}")
    print(f"2. Preprocess: python preprocess_to_curated.py --input {output_path}")
    print(f"3. Generate JSONs: python generate_series_json.py")
    
    return 0


if __name__ == "__main__":
    exit(main())
