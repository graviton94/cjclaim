import pandas as pd
from pathlib import Path
import tools.lag_analyzer as lag_analyzer
import datetime

def run_merge_and_lag_policy():
    """병합 + Lag 분석 + 정책 필터링 일괄 수행"""
    data_dir = Path('../data') if not (Path('data').exists()) else Path('data')
    raw_files = ['2021_raw.csv', '2022_raw.csv', '2023_raw.csv']
    yyyy_range = '2021_2023'
    merged_path = Path(f'artifacts/metrics/claims_{yyyy_range}_merged.csv')
    lag_stats_path = Path(f'artifacts/metrics/lag_stats_{yyyy_range}.csv')
    policy_out_path = Path(f'artifacts/metrics/candidates_claims_{yyyy_range}_with_lag_policy.csv')
    log_path = Path(f'artifacts/temp/pipeline_init_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    # 1. Merge all raw files
    dfs = [pd.read_csv(data_dir / f) for f in raw_files if (data_dir / f).exists()]
    if not dfs:
        with open(log_path, 'w', encoding='utf-8') as log:
            log.write(f"No raw CSV files found in '{data_dir.resolve()}'.\n")
        raise FileNotFoundError(f"No raw CSV files found in '{data_dir.resolve()}'.")
    df_all = pd.concat(dfs, ignore_index=True)

    # 2. Standardize columns and aggregate by event key
    # Use '발생일자' as the canonical date column for lag analysis
    if '발생일자' not in df_all.columns:
        raise RuntimeError("No date column '발생일자' found in merged raw files.")

    group_cols = ['발생일자', '플랜트', '제품범주2', '중분류', '제조일자']
    if 'count' in df_all.columns:
        df_all = df_all.groupby(group_cols, as_index=False)['count'].sum()
    df_all.to_csv(merged_path, index=False, encoding='utf-8-sig')

    # 3. Lag stats (call lag_analyzer.calculate_lag_stats)
    stats_df = lag_analyzer.calculate_lag_stats(df_all)
    stats_df.to_csv(lag_stats_path, index=False, encoding='utf-8-sig')

    # 4. Policy labeling (call lag_analyzer.label_and_filter)
    labeled_df, candidates_df = lag_analyzer.label_and_filter(df_all, stats_df)
    labeled_df.to_csv(policy_out_path, index=False, encoding='utf-8-sig')
    candidates_path = policy_out_path.parent / f'candidates_{policy_out_path.stem}.csv'
    candidates_df.to_csv(candidates_path, index=False, encoding='utf-8-sig')

    # 5. Logging
    with open(log_path, 'w', encoding='utf-8') as log:
        log.write(f"✅ Merged CSV saved: {merged_path}\n")
        log.write(f"✅ Lag stats saved: {lag_stats_path}\n")
        log.write(f"✅ Policy candidates saved: {policy_out_path}\n")
        log.write(f"✅ Candidates file saved: {candidates_path}\n")

if __name__ == "__main__":
    run_merge_and_lag_policy()
