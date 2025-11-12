"""
발생일자-제조일자 Lag 분석기 (제품범주2별)
- 2021-2023 Base 데이터에서 lag 통계 산출
- 월별 데이터 라벨링 (normal/borderline/extreme)
- Retrain 후보 시리즈 추출
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json


def calculate_lag_stats(df):
    """
    제품범주2별 lag 통계 계산
    
    Returns:
        DataFrame: product_cat2, mu, sigma, p90, p95, n, use_global
    """
    # 발생일자 - 제조일자 계산 (일 단위)
    df['lag_days'] = (pd.to_datetime(df['발생일자']) - pd.to_datetime(df['제조일자'])).dt.days
    
    # 음수 lag 제거 (제조일자 > 발생일자 → 잘못 접수된 케이스)
    df_invalid_negative = df[df['lag_days'] < 0]
    df_valid = df[df['lag_days'] >= 0].copy()
    
    print(f"총 레코드: {len(df):,}건")
    print(f"유효 레코드: {len(df_valid):,}건")
    print(f"⚠️  잘못 접수 제외: {len(df_invalid_negative):,}건 (제조일자 > 발생일자, 음수 lag)")
    
    # 제품범주2별 통계
    stats_list = []
    
    for product_cat2, group in df_valid.groupby('제품범주2'):
        lags = group['lag_days'].values
        n = len(lags)
        
        if n >= 30:  # 충분한 샘플
            mu = np.mean(lags)
            sigma = np.std(lags)
            p90 = np.percentile(lags, 90)
            p95 = np.percentile(lags, 95)
            use_global = False
        else:  # 소표본 - 글로벌 통계 사용
            use_global = True
            mu = None
            sigma = None
            p90 = None
            p95 = None
        
        stats_list.append({
            'product_cat2': product_cat2,
            'mu': mu,
            'sigma': sigma,
            'p90': p90,
            'p95': p95,
            'n': n,
            'use_global': use_global
        })
    
    stats_df = pd.DataFrame(stats_list)
    
    # 글로벌 통계 계산 (소표본용)
    global_mu = df_valid['lag_days'].mean()
    global_sigma = df_valid['lag_days'].std()
    global_p90 = df_valid['lag_days'].quantile(0.90)
    global_p95 = df_valid['lag_days'].quantile(0.95)
    
    # 소표본에 글로벌 통계 할당
    stats_df.loc[stats_df['use_global'] == True, 'mu'] = global_mu
    stats_df.loc[stats_df['use_global'] == True, 'sigma'] = global_sigma
    stats_df.loc[stats_df['use_global'] == True, 'p90'] = global_p90
    stats_df.loc[stats_df['use_global'] == True, 'p95'] = global_p95
    
    return stats_df


def label_and_filter(df, ref_stats):
    """
    월별 데이터에 lag 라벨 부여 및 필터링
    
    Args:
        df: 새로운 월 데이터
        ref_stats: 기준 lag 통계 DataFrame
    
    Returns:
        labeled_df: lag_class 컬럼 추가된 DataFrame
        candidates_df: retrain 후보 (normal + borderline만)
    """
    # lag 계산 (발생일자 - 제조일자)
    df['lag_days'] = (pd.to_datetime(df['발생일자']) - pd.to_datetime(df['제조일자'])).dt.days
    
    # 초기화: 모든 레코드 extreme으로 시작
    df['lag_class'] = 'extreme'
    
    # ⚠️ 음수 lag (제조일자 > 발생일자) → 잘못 접수된 케이스로 분류
    invalid_negative_mask = df['lag_days'] < 0
    df.loc[invalid_negative_mask, 'lag_class'] = 'invalid_negative'
    
    print(f"\n⚠️  잘못 접수 케이스: {invalid_negative_mask.sum():,}건 (제조일자 > 발생일자)")
    
    # 유효 레코드(lag >= 0)에 대해서만 라벨링
    valid_mask = df['lag_days'] >= 0
    print(f"\n⚠️  잘못 접수 케이스: {invalid_negative_mask.sum():,}건 (제조일자 > 발생일자)")
    
    # 유효 레코드(lag >= 0)에 대해서만 라벨링
    valid_mask = df['lag_days'] >= 0
    
    # 제품범주2별 라벨링
    for idx in df[valid_mask].index:
        product_cat2 = df.loc[idx, '제품범주2']
        lag = df.loc[idx, 'lag_days']
        
        # 해당 제품범주2의 통계 찾기
        stat = ref_stats[ref_stats['product_cat2'] == product_cat2]
        
        if len(stat) == 0:
            # 통계 없으면 글로벌 통계 사용
            global_stat = ref_stats[ref_stats['use_global'] == True].iloc[0]
            mu = global_stat['mu']
            sigma = global_stat['sigma']
        else:
            mu = stat.iloc[0]['mu']
            sigma = stat.iloc[0]['sigma']
        
        # μ+σ 기준 라벨링
        if lag <= mu + sigma:
            df.loc[idx, 'lag_class'] = 'normal'
        elif lag <= mu + 2 * sigma:
            df.loc[idx, 'lag_class'] = 'borderline'
        else:
            df.loc[idx, 'lag_class'] = 'extreme'
    
    # 라벨 분포 출력
    print("\n📊 Lag 라벨 분포:")
    label_counts = df['lag_class'].value_counts()
    print(label_counts)
    print(f"\n비율:")
    label_pct = df['lag_class'].value_counts(normalize=True) * 100
    for label, pct in label_pct.items():
        print(f"  {label:20s}: {pct:5.1f}%")
    
    # normal + borderline만 학습 후보 (invalid_negative와 extreme 제외)
    candidates = df[df['lag_class'].isin(['normal', 'borderline'])].copy()
    
    print(f"\n✅ 학습 후보: {len(candidates):,}건 / {len(df):,}건 ({len(candidates)/len(df)*100:.1f}%)")
    print(f"   - Normal:     {(df['lag_class'] == 'normal').sum():,}건")
    print(f"   - Borderline: {(df['lag_class'] == 'borderline').sum():,}건")
    print(f"\n❌ 학습 제외: {len(df) - len(candidates):,}건")
    print(f"   - Invalid (음수 lag): {(df['lag_class'] == 'invalid_negative').sum():,}건")
    print(f"   - Extreme (μ+2σ 초과): {(df['lag_class'] == 'extreme').sum():,}건")
    
    # weight 할당
    candidates['sample_weight'] = candidates['lag_class'].map({
        'normal': 1.0,
        'borderline': 0.5
    })
    
    return df, candidates


def main():
    parser = argparse.ArgumentParser(description='Lag 분석 및 라벨링')
    parser.add_argument('--input', required=True, help='입력 CSV 파일')
    parser.add_argument('--ref', help='기준 lag 통계 CSV (라벨링 모드)')
    parser.add_argument('--out', help='출력 lag 통계 CSV (통계 산출 모드)')
    parser.add_argument('--policy-out', help='라벨링 결과 CSV (라벨링 모드)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Lag Analyzer - 제품범주2별 발생일자-제조일자 lag 분석")
    print("=" * 80)
    
    # 입력 파일 읽기
    input_path = Path(args.input)
    print(f"\n입력: {input_path}")
    
    # 인코딩 자동 감지 시도
    encodings = ['cp949', 'euc-kr', 'utf-8']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(input_path, encoding=enc)
            print(f"인코딩: {enc}")
            break
        except:
            continue
    
    if df is None:
        raise ValueError(f"파일을 읽을 수 없습니다: {input_path}")
    
    print(f"레코드: {len(df):,}건")
    
    # 컬럼명 고정 매핑 (추가 컬럼 허용)
    expected_cols = ['발생일자', '중분류', '플랜트', '제품범주2', '제조일자', 'count']
    
    # 컬럼 수가 정확히 일치하면 표준화
    if len(df.columns) == len(expected_cols):
        df.columns = expected_cols
        print(f"컬럼명 표준화 완료: {df.columns.tolist()}")
    # 더 많은 컬럼이 있으면 (year, month 등 추가 컬럼 포함)
    elif len(df.columns) > len(expected_cols):
        # 필수 컬럼이 모두 있는지 확인
        missing_required = [col for col in expected_cols if col not in df.columns]
        if not missing_required:
            print(f"✅ 필수 컬럼 확인 완료 (추가 컬럼 포함: {[c for c in df.columns if c not in expected_cols]})")
        else:
            # 첫 6개가 필수 컬럼 순서라고 가정
            df.columns = expected_cols + list(df.columns[len(expected_cols):])
            print(f"컬럼명 표준화 완료 (추가 컬럼 보존): {df.columns.tolist()}")
    else:
        print(f"경고: 예상 컬럼 수({len(expected_cols)})보다 적습니다({len(df.columns)}).")
        print(f"현재 컬럼: {df.columns.tolist()}")
    
    # count를 클레임건수로 사용
    if 'count' in df.columns and '클레임건수' not in df.columns:
        df['클레임건수'] = df['count']

    # 필수 컬럼 확인 (발생일자 사용)
    required_cols = ['제조일자', '발생일자', '제품범주2']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"\n현재 컬럼: {df.columns.tolist()}")
        raise ValueError(f"필수 컬럼 누락: {missing}")
    
    # 모드 분기
    if args.ref is None:
        # === 통계 산출 모드 ===
        print("\n[모드] 통계 산출")
        
        stats_df = calculate_lag_stats(df)
        
        print("\n제품범주2별 Lag 통계:")
        print(stats_df.to_string())
        
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            stats_df.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"\n✅ 통계 저장: {out_path}")
        
        # 요약 출력
        print("\n" + "=" * 80)
        print("요약 통계:")
        print(f"제품범주2 수: {len(stats_df)}")
        print(f"소표본(<30) 제품: {stats_df['use_global'].sum()}개")
        print(f"\n전체 평균 lag: {stats_df['mu'].mean():.1f}일")
        print(f"전체 평균 sigma: {stats_df['sigma'].mean():.1f}일")
        print("=" * 80)
    
    else:
        # === 라벨링 모드 ===
        print("\n[모드] 라벨링 및 필터링")
        
        ref_path = Path(args.ref)
        print(f"기준 통계: {ref_path}")
        
        ref_stats = pd.read_csv(ref_path)
        print(f"기준 제품범주2: {len(ref_stats)}개")
        
        labeled_df, candidates_df = label_and_filter(df, ref_stats)
        
        print(f"\n학습 후보: {len(candidates_df):,}건 / {len(df):,}건 ({len(candidates_df)/len(df)*100:.1f}%)")
        
        if args.policy_out:
            out_path = Path(args.policy_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 전체 라벨링 결과 저장
            labeled_df.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"\n✅ 라벨링 결과 저장: {out_path}")
            
            # 후보 목록도 별도 저장
            candidates_path = out_path.parent / f"candidates_{out_path.stem}.csv"
            candidates_df.to_csv(candidates_path, index=False, encoding='utf-8-sig')
            print(f"✅ 학습 후보 저장: {candidates_path}")
        
        # 시리즈별 요약 (표준화된 컬럼명 사용)
        series_summary = candidates_df.groupby(['플랜트', '제품범주2', '중분류']).agg({
            'count': 'sum',
            'sample_weight': 'mean',
            'lag_class': lambda x: (x == 'normal').sum()
        }).reset_index()
        series_summary.columns = ['플랜트', '제품범주2', '중분류', '총클레임', '평균가중치', 'normal건수']
        
        print("\n시리즈별 요약 (상위 10개):")
        print(series_summary.nlargest(10, '총클레임').to_string(index=False))


if __name__ == '__main__':
    main()
