"""
월별 데이터 Lag 필터링 및 검증
"""
import pandas as pd
import subprocess
from pathlib import Path


def filter_monthly_data(
    input_csv: str,
    year: int,
    month: int,
    lag_stats_path: str = "artifacts/metrics/lag_stats_from_raw.csv",
    output_dir: str = "artifacts/incremental"
):
    """
    월별 데이터를 Lag 정책으로 필터링
    
    Returns:
    --------
    dict: {
        'total': 전체 레코드 수,
        'normal': Normal 레코드 수,
        'borderline': Borderline 레코드 수,
        'extreme': Extreme 레코드 수,
        'candidates_file': 학습 후보 파일 경로,
        'filtered_file': 전체 라벨링 파일 경로
    }
    """
    
    output_path = Path(output_dir) / f"{year}_{month:02d}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 출력 파일 경로
    filtered_file = output_path / f"filtered_{year}_{month:02d}.csv"
    candidates_file = output_path / f"candidates_{year}_{month:02d}.csv"
    
    # lag_analyzer 호출
    print(f"  Lag 분석 실행...")
    cmd = [
        "python", "tools/lag_analyzer.py",
        "--input", input_csv,
        "--ref", lag_stats_path,
        "--policy-out", str(filtered_file)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Lag 분석 실패: {result.stderr}")
    
    # 결과 파일 로드
    df_filtered = pd.read_csv(filtered_file, encoding='utf-8-sig')
    
    # 통계 계산
    stats = {
        'total': len(df_filtered),
        'normal': (df_filtered['lag_class'] == 'normal').sum(),
        'borderline': (df_filtered['lag_class'] == 'borderline').sum(),
        'extreme': (df_filtered['lag_class'] == 'extreme').sum() + 
                   (df_filtered['lag_class'] == 'extreme_negative').sum(),
        'candidates_file': str(candidates_file),
        'filtered_file': str(filtered_file)
    }
    
    print(f"  ✅ 필터링 완료:")
    print(f"     Total:      {stats['total']:,}건")
    print(f"     Normal:     {stats['normal']:,}건 ({stats['normal']/stats['total']*100:.1f}%)")
    print(f"     Borderline: {stats['borderline']:,}건 ({stats['borderline']/stats['total']*100:.1f}%)")
    print(f"     Extreme:    {stats['extreme']:,}건 ({stats['extreme']/stats['total']*100:.1f}%)")
    print(f"     학습 후보:   {stats['normal'] + stats['borderline']:,}건")
    
    return stats


if __name__ == '__main__':
    # 테스트
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    args = parser.parse_args()
    
    stats = filter_monthly_data(args.input, args.year, args.month)
    print(f"\n결과: {stats}")
