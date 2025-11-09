"""
예측 생성 디버깅 스크립트
"""
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# 첫 번째 모델 선택
model_path = Path("artifacts/models/base_monthly/공주공장_된장가정용_곰팡이.pkl")

print(f"모델 로드: {model_path}")
with open(model_path, 'rb') as f:
    model_info = pickle.load(f)

series_id = model_info.get('series_id')
print(f"Series ID: {series_id}")

# JSON 파일명 생성
safe_filename = (series_id.replace('/', '_').replace('\\', '_').replace(':', '_')
                .replace('|', '_').replace('?', '_').replace('*', '_')
                .replace('<', '_').replace('>', '_').replace('"', '_'))
json_path = Path("data/features") / f"{safe_filename}.json"

print(f"JSON 경로: {json_path}")
print(f"JSON 존재: {json_path.exists()}")

if json_path.exists():
    with open(json_path, 'r', encoding='utf-8') as f:
        series_data = json.load(f)
    
    print(f"JSON 데이터 키: {series_data.keys()}")
    print(f"레코드 수: {len(series_data.get('data', []))}")
    
    # 학습 데이터 추출
    data_records = series_data.get('data', [])
    df = pd.DataFrame(data_records)
    print(f"DataFrame 컬럼: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")
    
    # 2021-2023 필터링
    df_train = df[(df['year'] >= 2021) & (df['year'] <= 2023)].copy()
    print(f"2021-2023 데이터: {len(df_train)} rows")
    
    if len(df_train) >= 12:  # 최소 12개월
        y = df_train['claim_count'].values
        print(f"y 길이: {len(y)}")
        print(f"y 샘플: {y[:5]}")
        
        # 모델 재구성
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        model_spec = model_info.get('model_spec', {})
        print(f"모델 스펙: {model_spec}")
        
        order = model_spec.get('order', (1, 0, 1))
        seasonal_order = model_spec.get('seasonal_order', (1, 0, 1, 12))
        
        print(f"Order: {order}")
        print(f"Seasonal order: {seasonal_order}")
        
        try:
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=model_spec.get('enforce_stationarity', False),
                enforce_invertibility=model_spec.get('enforce_invertibility', False)
            )
            
            # 저장된 파라미터
            params = np.array(model_info.get('params', []))
            print(f"Params 길이: {len(params)}")
            print(f"Params 샘플: {params[:5]}")
            
            # Smoothing
            fitted_model = model.smooth(params)
            print("모델 스무딩 성공!")
            
            # 예측
            horizon = 6
            forecast = fitted_model.forecast(steps=horizon)
            forecast = np.maximum(forecast, 0)
            
            print(f"예측 성공! 길이: {len(forecast)}")
            print(f"예측값: {forecast}")
            
            # 신뢰구간
            forecast_result = fitted_model.get_forecast(steps=horizon)
            conf_int = forecast_result.conf_int(alpha=0.05)
            print(f"신뢰구간: {conf_int}")
            
            # 주차 계산
            year_int = 2024
            month_int = 1
            first_day = datetime(year_int, month_int, 1)
            
            weeks = []
            for i in range(horizon):
                date = first_day + timedelta(weeks=i)
                iso_year, iso_week, _ = date.isocalendar()
                weeks.append({'year': iso_year, 'week': iso_week})
            
            print(f"주차: {weeks}")
            
            print("\n✅ 전체 프로세스 성공!")
            
        except Exception as e:
            import traceback
            print(f"\n❌ 모델 재구성/예측 실패:")
            print(f"에러: {e}")
            print(traceback.format_exc())
    else:
        print(f"⚠️ 데이터 부족 (최소 12 필요, 현재 {len(df_train)})")
else:
    print(f"❌ JSON 파일 없음")
