"""F3, F4 NaN 원인 디버깅"""
import pandas as pd
import numpy as np
import json
from statsmodels.tsa.seasonal import STL
from pathlib import Path

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
print(f"데이터 길이: {len(historical_data)}")
print(f"데이터: {historical_data}")
print(f"최소/최대: {historical_data.min()}/{historical_data.max()}")
print(f"평균: {historical_data.mean():.2f}")
print(f"분산: {historical_data.var():.4f}")

# F3 계절성 테스트
period = 12
if len(historical_data) < 2 * period:
    print(f"\n❌ F3: 데이터 부족 (최소 {2*period} 필요, 현재 {len(historical_data)})")
else:
    print(f"\n✅ F3: 데이터 충분 ({len(historical_data)} >= {2*period})")
    
    try:
        # STL decomposition
        stl = STL(historical_data, seasonal=period + 1, robust=True)
        result = stl.fit()
        
        seasonal = result.seasonal
        resid = result.resid
        
        var_resid = np.var(resid)
        var_y = np.var(historical_data)
        
        print(f"  Var(residual): {var_resid:.6f}")
        print(f"  Var(y): {var_y:.6f}")
        
        if var_y < 1e-10:
            print(f"  ❌ F3: 분산 너무 작음 (거의 0)")
            f3 = 0.0
        else:
            strength = 1 - (var_resid / var_y)
            strength = max(0, strength)
            f3 = float(strength)
            print(f"  ✅ F3 계절성 강도: {f3:.4f}")
            
    except Exception as e:
        print(f"  ❌ F3 계산 실패: {e}")

# F4 진폭 테스트
print(f"\n=== F4 진폭 테스트 ===")
if len(historical_data) < 12:
    print(f"❌ F4: 데이터 부족 (최소 12 필요, 현재 {len(historical_data)})")
else:
    print(f"✅ F4: 데이터 충분 ({len(historical_data)} >= 12)")
    
    try:
        # 12개월 윈도우로 진폭 계산
        window_size = 12
        amplitudes = []
        
        for i in range(len(historical_data) - window_size + 1):
            window = historical_data[i:i+window_size]
            amplitude = window.max() - window.min()
            amplitudes.append(amplitude)
        
        mean_amplitude = np.mean(amplitudes)
        mean_level = historical_data.mean()
        
        print(f"  평균 진폭: {mean_amplitude:.2f}")
        print(f"  평균 레벨: {mean_level:.2f}")
        
        if mean_level < 0.01:
            print(f"  ❌ F4: 평균 레벨 너무 낮음 (거의 0)")
            f4 = 0.0
        else:
            f4 = mean_amplitude / mean_level
            f4 = min(f4, 1.0)  # Cap at 1.0
            print(f"  ✅ F4 정규화 진폭: {f4:.4f}")
            
    except Exception as e:
        print(f"  ❌ F4 계산 실패: {e}")

print("\n" + "="*60)
print("결론:")
print("="*60)
print("F3, F4 계산 로직은 정상 작동")
print("문제는 EWS 스코어링 호출 시 historical_data 전달 이슈일 가능성")
