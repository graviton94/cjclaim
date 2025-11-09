import pandas as pd

df = pd.read_csv('artifacts/metrics/ews_scores_2024_01_v2.csv')

print('='*60)
print('개선된 EWS 스코어 분석')
print('='*60)

print('\n=== F3, F4 유효 데이터 ===')
print(f'F3 유효: {(~df["f3_season"].isna()).sum()}/{len(df)} ({(~df["f3_season"].isna()).sum()/len(df)*100:.1f}%)')
print(f'F4 유효: {(~df["f4_ampl"].isna()).sum()}/{len(df)} ({(~df["f4_ampl"].isna()).sum()/len(df)*100:.1f}%)')

print(f'\n=== F3 계절성 통계 ===')
print(df['f3_season'].describe())

print(f'\n=== F4 진폭 통계 ===')
print(df['f4_ampl'].describe())

print(f'\n=== EWS 후보 (S>=0.4, A>=0.3) ===')
print(f'Valid candidates: {df["candidate"].sum()}개')

print(f'\n=== 개선 전 vs 개선 후 ===')
print('개선 전: F3=0%, F4=0% (모두 NaN)')
print(f'개선 후: F3={(~df["f3_season"].isna()).sum()/len(df)*100:.1f}%, F4={(~df["f4_ampl"].isna()).sum()/len(df)*100:.1f}%')

print(f'\n=== EWS 레벨 분포 ===')
print(df['level'].value_counts())
