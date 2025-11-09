"""EWS 스코어 결과 분석"""
import pandas as pd
import numpy as np

df = pd.read_csv('artifacts/metrics/ews_scores_2024_01.csv')

print('=' * 80)
print('EWS 스코어 파일 분석')
print('=' * 80)

print(f'\n=== 기본 정보 ===')
print(f'총 시리즈 수: {len(df)}')
print(f'컬럼: {df.columns.tolist()}')

print(f'\n=== 상위 20개 (EWS Score 기준) ===')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(df.head(20)[['rank', 'series_id', 'ews_score', 'level', 'f1_ratio', 'f2_conf', 'f3_season', 'f4_ampl', 'f5_inflect']].to_string())

print(f'\n=== EWS 레벨 분포 ===')
print(df['level'].value_counts())

print(f'\n=== EWS 스코어 통계 ===')
print(df['ews_score'].describe())

print(f'\n=== 5-Factor 유효 데이터 비율 ===')
print(f'F1 (증가율) 유효: {(~df["f1_ratio"].isna()).sum()}/{len(df)} ({(~df["f1_ratio"].isna()).sum()/len(df)*100:.1f}%)')
print(f'F2 (신뢰도) 유효: {(~df["f2_conf"].isna()).sum()}/{len(df)} ({(~df["f2_conf"].isna()).sum()/len(df)*100:.1f}%)')
print(f'F3 (계절성) 유효: {(~df["f3_season"].isna()).sum()}/{len(df)} ({(~df["f3_season"].isna()).sum()/len(df)*100:.1f}%)')
print(f'F4 (진폭) 유효: {(~df["f4_ampl"].isna()).sum()}/{len(df)} ({(~df["f4_ampl"].isna()).sum()/len(df)*100:.1f}%)')
print(f'F5 (변곡점) 유효: {(~df["f5_inflect"].isna()).sum()}/{len(df)} ({(~df["f5_inflect"].isna()).sum()/len(df)*100:.1f}%)')

print(f'\n=== F3, F4 NaN 원인 분석 ===')
print(f'F3 NaN: {df["f3_season"].isna().sum()}/{len(df)} ({df["f3_season"].isna().sum()/len(df)*100:.1f}%)')
print(f'F4 NaN: {df["f4_ampl"].isna().sum()}/{len(df)} ({df["f4_ampl"].isna().sum()/len(df)*100:.1f}%)')

print(f'\n=== 월별 예측 vs 6개월 예측 문제 ===')
print('현재 구현:')
print('  - forecast_2024_01.parquet: 2024년 1월부터 6개월 예측 (1-6월)')
print('  - EWS 스코어링: 전체 6개월 예측 사용')
print('  - 문제점: 월별로 위험도가 다를 수 있음 (1월 안전, 3월 위험 등)')

print(f'\n=== HIGH/MEDIUM/LOW 분포 ===')
if 'level' in df.columns:
    high = (df['level'] == 'HIGH').sum()
    medium = (df['level'] == 'MEDIUM').sum()
    low = (df['level'] == 'LOW').sum()
    low_conf = (df['level'] == 'LOW_CONF').sum()
    print(f'HIGH: {high} ({high/len(df)*100:.1f}%)')
    print(f'MEDIUM: {medium} ({medium/len(df)*100:.1f}%)')
    print(f'LOW: {low} ({low/len(df)*100:.1f}%)')
    print(f'LOW_CONF: {low_conf} ({low_conf/len(df)*100:.1f}%)')

print(f'\n=== 증가율(F1) 분포 ===')
print(df['f1_ratio'].describe())
print(f'F1 >= 1.5 (50%+ 증가): {(df["f1_ratio"] >= 1.5).sum()}개')
print(f'F1 >= 2.0 (2배 증가): {(df["f1_ratio"] >= 2.0).sum()}개')
print(f'F1 >= 3.0 (3배 증가): {(df["f1_ratio"] >= 3.0).sum()}개')

print(f'\n=== 신뢰도(F2) 분포 ===')
print(df['f2_conf'].describe())
print(f'F2 < 0.3 (낮은 신뢰도): {(df["f2_conf"] < 0.3).sum()}개')
print(f'F2 >= 0.5 (중간 이상): {(df["f2_conf"] >= 0.5).sum()}개')
print(f'F2 >= 0.7 (높은 신뢰도): {(df["f2_conf"] >= 0.7).sum()}개')

print('\n' + '=' * 80)
print('허점 분석')
print('=' * 80)

print('\n1. F3(계절성), F4(진폭) 대부분 NaN')
print('   - 원인: 월별 데이터(36개월)로는 계절성 패턴 감지 어려움')
print('   - 해결: 주별 데이터 또는 더 긴 히스토리 필요')

print('\n2. 6개월 예측이지만 월별 위험도 미분석')
print('   - 현재: 6개월 평균으로 EWS 계산')
print('   - 문제: 특정 월에만 급증하는 패턴 놓칠 수 있음')
print('   - 해결: 월별 EWS 스코어 계산 또는 시간대별 가중치')

print('\n3. LOW_CONF 시리즈 처리')
low_conf_count = (df['level'] == 'LOW_CONF').sum()
print(f'   - LOW_CONF: {low_conf_count}개 ({low_conf_count/len(df)*100:.1f}%)')
print('   - 문제: 신뢰도 낮아도 증가율 높으면 EWS 1.0')
print('   - 해결: 신뢰도 임계값 적용 또는 별도 카테고리')

print('\n4. EWS 스코어 대부분 1.0')
score_1 = (df['ews_score'] >= 0.99).sum()
print(f'   - 스코어 1.0: {score_1}개 ({score_1/len(df)*100:.1f}%)')
print('   - 문제: 변별력 부족, 우선순위 판단 어려움')
print('   - 해결: 가중치 재조정 또는 스코어링 함수 수정')

print('\n5. 계절성/진폭 가중치 낭비')
print('   - F3, F4가 대부분 NaN이면 가중치 0.45 (30%+15%) 사용 안됨')
print('   - 해결: 월별 데이터에 맞는 factor 추가 또는 가중치 재분배')

print('\n' + '=' * 80)
print('권장 개선사항')
print('=' * 80)

print('\n1. 월별 위험도 추적')
print('   - 2024-01, 2024-02, ..., 2024-06 각각의 EWS 계산')
print('   - "어느 달에 위험한가?" 정보 제공')

print('\n2. 신뢰도 가중 적용')
print('   - F2 < 0.3이면 EWS 스코어에 페널티')
print('   - 또는 HIGH_UNCERTAIN 별도 카테고리')

print('\n3. 월별 데이터 특화 factor')
print('   - F3 계절성 → 전년동월 비교 (YoY)')
print('   - F4 진폭 → 월별 변동계수 (CV)')
print('   - F6 추가: 연속 증가 개월 수')

print('\n4. Top-K 선정 기준 명확화')
print('   - EWS >= 0.7 AND F2 >= 0.5 AND F1 >= 1.5')
print('   - 또는 각 factor 임계값 통과한 시리즈만')
