# 월별 증분학습 워크플로우

## 📋 **전체 프로세스**

```
[GUI 업로드]
     ↓
1. 월별 데이터 수신 (발생일자 기준)
     ↓
2. Lag 필터링 (Normal-Lag만)
     ↓
3. 기존 예측과 비교
     ↓
4. KPI 게이트 체크
     ↓
5. 모델 재학습 (append_fit)
     ↓
6. 결과 기록
```

---

## 🚀 **사용 방법**

### **기본 실행**
```bash
python process_monthly_incremental.py \
  --upload data/uploaded/claims_2024_01.csv \
  --year 2024 \
  --month 1
```

### **출력 결과**

```
artifacts/incremental/2024_01/
├── filtered_2024_01.csv          # 전체 라벨링 데이터
├── candidates_2024_01.csv        # 학습 후보 (Normal+Borderline)
└── models/                       # 업데이트된 모델 (TODO)

logs/
├── predict_vs_actual_2024_01.json    # 예측-실측 비교
└── incremental/
    └── summary_2024_01.json          # 전체 요약
```

---

## 📊 **Step별 상세**

### **Step 1: 데이터 로드**
- GUI에서 업로드한 CSV 읽기
- 발생일자 기준 1개월 데이터

### **Step 2: Lag 필터링**
```python
# tools/filter_monthly_data.py
filter_stats = filter_monthly_data(
    input_csv="uploaded.csv",
    year=2024,
    month=1,
    lag_stats_path="artifacts/metrics/lag_stats_from_raw.csv"
)

# 결과
{
    'total': 1500,
    'normal': 1200,      # 80% (weight=1.0)
    'borderline': 200,   # 13% (weight=0.5)
    'extreme': 100,      # 7% (제외)
    'candidates_file': '...',
    'filtered_file': '...'
}
```

### **Step 3: 예측-실측 비교**
```python
# tools/compare_forecast_actual.py
series_metrics = compare_forecast_vs_actual(
    actual_file="candidates_2024_01.csv",
    forecast_file="artifacts/forecasts/2024/forecast_2024_01.parquet",
    year=2024,
    month=1
)

# 시리즈별 메트릭
{
    "진천BC|백미|미생물": {
        "MAPE": 15.2,      # %
        "Bias": -0.03,     # 과소예측
        "MAE": 3.5,
        "RMSE": 5.2,
        "R2": 0.85
    },
    ...
}
```

### **Step 4: KPI 게이트**
```python
# KPI 기준
MAPE < 20%    # 평균 절대 오차율
|Bias| < 0.05 # 편향

# 결과
if kpi_pass:
    → 정상 학습 진행
else:
    → Reconcile 필요 (Bias Map, Seasonal Adj, Optuna)
```

### **Step 5: 모델 재학습**
```python
# TODO: append_fit 구현 예정
# 각 시리즈별:
# 1. 기존 model.pkl 로드
# 2. start_params 추출
# 3. 새 데이터로 재적합 (sample_weight 적용)
# 4. 업데이트된 model.pkl 저장
```

### **Step 6: 결과 기록**
```json
{
  "year": 2024,
  "month": 1,
  "timestamp": "2024-01-31T23:59:59",
  "filter_stats": {
    "total": 1500,
    "normal": 1200,
    "borderline": 200,
    "extreme": 100
  },
  "kpi_pass": true,
  "series_count": 289,
  "forecast_file": "...",
  "candidates_file": "..."
}
```

---

## 🔄 **증분학습 흐름**

```
2024-01 데이터 도착
    ↓
Lag 필터링 (95% 통과)
    ↓
예측 vs 실제 비교
    ↓
MAPE 15%, Bias -0.03 (통과)
    ↓
모델 재학습 (append_fit)
    ↓
다음 달 예측 갱신
```

---

## ✅ **완성된 파일**

| 파일 | 기능 |
|------|------|
| `process_monthly_incremental.py` | 메인 파이프라인 |
| `tools/filter_monthly_data.py` | Lag 필터링 |
| `tools/compare_forecast_actual.py` | 예측-실측 비교 |

---

## 🎯 **다음 구현 필요**

1. **append_fit 로직** (Step 5)
   - forecasting.py 수정
   - start_params 활용
   - sample_weight 지원

2. **Reconcile 보정** (KPI 미달 시)
   - Bias Map
   - Seasonal Recalibration
   - Optuna 하이퍼파라미터 튜닝

3. **GUI 연동**
   - Streamlit 업로드 버튼
   - process_monthly_incremental.py 호출
   - 진행 상황 표시
