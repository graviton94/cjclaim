"""개선된 F3, F4 테스트"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
sys.path.append('.')

from src.ews_scoring_v2 import EWSScorer

# 샘플 시리즈 로드
json_path = Path("data/features/공주공장_된장가정용_곰팡이.json")

with open(json_path, 'r', encoding='utf-8') as f:
    series_data = json.load(f)

series_id = series_data['series_id']
df = pd.DataFrame(series_data['data'])

# 2021-2023 필터링
df_train = df[(df['year'] >= 2021) & (df['year'] <= 2023)].copy()
historical_data = df_train['claim_count'].values

print(f"Series: {series_id}")
print(f"데이터: {len(historical_data)}개월 (2021-2023)")
print(f"Values: {historical_data[:12]} ...")  # 첫 12개월만

# EWSScorer 인스턴스 생성
scorer = EWSScorer()

# F3 계절성 테스트
print("\n=== F3 계절성 (개선 버전) ===")
f3 = scorer.calculate_f3_seasonality(historical_data, period=12)
if np.isnan(f3):
    print(f"❌ F3: NaN")
else:
    print(f"✅ F3 계절성 강도: {f3:.4f}")

# F4 진폭 테스트  
print("\n=== F4 진폭 (개선 버전) ===")
f4 = scorer.calculate_f4_amplitude(historical_data, period=12)
if np.isnan(f4):
    print(f"❌ F4: NaN")
else:
    print(f"✅ F4 정규화 진폭: {f4:.4f}")

# 여러 시리즈 테스트
print("\n" + "="*60)
print("여러 시리즈 테스트")
print("="*60)

json_files = list(Path("data/features").glob("*.json"))[:10]  # 첫 10개만
results = []

for json_file in json_files:
    if json_file.name.startswith('_'):
        continue
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['data'])
        df_train = df[(df['year'] >= 2021) & (df['year'] <= 2023)]
        
        if len(df_train) < 24:
            continue
        
        historical = df_train['claim_count'].values
        
        f3 = scorer.calculate_f3_seasonality(historical, period=12)
        f4 = scorer.calculate_f4_amplitude(historical, period=12)
        
        results.append({
            'series_id': data['series_id'],
            'n': len(historical),
            'mean': historical.mean(),
            'f3': f3,
            'f4': f4
        })
    except:
        continue

df_results = pd.DataFrame(results)
print(f"\n처리된 시리즈: {len(df_results)}개")
print(f"F3 유효: {(~df_results['f3'].isna()).sum()}개 ({(~df_results['f3'].isna()).sum()/len(df_results)*100:.1f}%)")
print(f"F4 유효: {(~df_results['f4'].isna()).sum()}개 ({(~df_results['f4'].isna()).sum()/len(df_results)*100:.1f}%)")

print("\n샘플 결과:")
print(df_results.head(10).to_string())
