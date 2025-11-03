# 🔄 Batch 파이프라인 후속 설계 – 롤링 백테스트 & 동적 튜닝 전략

## 📍 구현 순서

1️⃣ **Baseline 검증 (잔차·지표 확인)** → `tools/validate_baseline.py`
2️⃣ **Rolling 백테스트로 기준선 확립** → `batch.py roll`
3️⃣ **Bias/Seasonal 경량 보정** → `src/reconcile.py` 확장
4️⃣ **Optuna 기반 조건부 재학습(튜닝)** → `tools/run_optuna.py`
5️⃣ **Reconcile 통합 보고** → `pipeline_reconcile.py` 확장

---

## 1️⃣ Baseline 진단 (학습 직후)

### 구현 파일: `tools/validate_baseline.py`

| 항목 | 검증 내용 | 산출물 |
|------|-----------|---------|
| 잔차 진단 | Ljung–Box(p<0.05), ACF, 정규성 | `reports/residual_report.md` |
| 기준 지표 | 연도·시리즈별 MAPE/MASE/Bias | `artifacts/metrics/metrics_baseline.parquet` |
| 폴백율 | Seasonal Naive 비율 | `artifacts/metrics/fallback_summary.csv` |

**목적:** Optuna 전에 문제 시리즈를 정량적으로 특정

```python
# 예시 코드 구조
def validate_baseline(artifacts_path):
    # 1. 잔차 분석
    # 2. MAPE/MASE/Bias 계산
    # 3. 폴백 시리즈 식별
    pass
```

---

## 2️⃣ Rolling 백테스트 (기준선 확립)

### 구현: `batch.py` 확장

```bash
python batch.py roll --start 2020 --end 2024
```

| 검증 항목 | 기준 | 산출 |
|-----------|------|------|
| 연간 MAPE | ≤ 0.20 | `tuning_candidates.csv` |
| Bias | ±5%p | |
| 분기별 이동 MAPE | 하락 시 플래그 | `bias_trend.parquet` |

**결과:** Top-worse 20% 시리즈 자동 선별 → 튜닝 대상 결정

```python
def rolling_backtest(start_year, end_year):
    for year in range(start_year, end_year + 1):
        # 해당 연도까지 데이터로 학습
        # 다음 기간 예측
        # 메트릭 계산
        pass
```

---

## 3️⃣ 경량 보정 (Bias/Seasonal)

### ① Bias Map 보정 (Week-map)

**구현 위치:** `src/reconcile.py`

```python
def apply_bias_correction(y_pred, y_true, week):
    """주차별 편향 보정"""
    bias_map = (y_true - y_pred).groupby('week').mean()
    y_pred_adj = y_pred + bias_map.get(week, 0.0)
    return y_pred_adj
```

### ② Seasonal Recalibration

* 최근 2년 데이터로 **SARIMAX 계절 성분만 재추정**
* 대안: Holt–Winters Smoothing

```python
def seasonal_recalibration(series_data, last_n_years=2):
    """계절성 재보정"""
    recent_data = series_data[series_data['year'] >= max_year - last_n_years]
    # SARIMAX seasonal component 재추정
    pass
```

### ③ Changepoint-aware Hold

* ruptures 감지 구간은 폴백 모델 고정

```python
def detect_changepoints_ruptures(y):
    """변화점 감지"""
    import ruptures as rpt
    algo = rpt.Pelt(model="rbf").fit(y)
    result = algo.predict(pen=10)
    return result
```

**효과:** 저비용(≈10초/시리즈)으로 절반 이상 편차 개선

---

## 4️⃣ Optuna 조건부 튜닝 (문제 시리즈만)

### 구현 파일: `tools/run_optuna.py`

| 항목 | 내용 |
|------|------|
| 대상 | MAPE>0.20 또는 Bias>0.05 시리즈 |
| 탐색공간 | p,d,q,P,D,Q ∈ [0,2], s=52 |
| 목표함수 | Rolling MASE 최소화 |
| 시간제한 | ≤ 10분/시리즈, trial ≤ 40 |
| 병렬 | ProcessPoolExecutor, max 6 workers |

```python
import optuna
from statsmodels.tsa.statespace.sarimax import SARIMAX

def objective(trial, y):
    """Optuna 목표 함수"""
    p = trial.suggest_int('p', 0, 2)
    d = trial.suggest_int('d', 0, 2)
    q = trial.suggest_int('q', 0, 2)
    P = trial.suggest_int('P', 0, 2)
    D = trial.suggest_int('D', 0, 2)
    Q = trial.suggest_int('Q', 0, 2)
    
    try:
        model = SARIMAX(
            y, 
            order=(p, d, q), 
            seasonal_order=(P, D, Q, 52)
        ).fit(disp=False)
        
        fc = model.forecast(steps=13)
        return compute_mase(y[-13:], fc)
    except:
        return float('inf')

def tune_series(series_id, y, timeout=600, n_trials=40):
    """시리즈별 튜닝 실행"""
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, y),
        n_trials=n_trials,
        timeout=timeout
    )
    return study.best_params
```

**실행:**
```bash
python tools/run_optuna.py --candidates artifacts/metrics/tuning_candidates.csv --timeout 600 --n-trials 40
```

---

## 5️⃣ Reconcile 통합 흐름

### 구현 순서

1️⃣ Bias Map → 2️⃣ Seasonal Recalib → 3️⃣ Optuna Refit(조건부)

| 단계 | 트리거 | 결과 | 로그 플래그 |
|------|--------|------|-------------|
| Bias 보정 | MAPE>0.10 | `bias_intercept` 저장 | `bias_adj=True` |
| 재계절화 | MAPE>0.20 | 모델 갱신 | `reseason=True` |
| Optuna | 개선 불충분 | 모델 교체 | `tuned=True` |

**구현 위치:** `pipeline_reconcile.py`

```python
def reconcile_with_tuning(forecast_df, actual_df, artifacts):
    """통합 보정 파이프라인"""
    results = []
    
    for series_id in forecast_df['series_id'].unique():
        series_forecast = forecast_df[forecast_df['series_id'] == series_id]
        series_actual = actual_df[actual_df['series_id'] == series_id]
        
        # 1. 메트릭 계산
        mape = compute_mape(series_actual['y'], series_forecast['yhat'])
        
        # 2. Bias 보정
        if mape > 0.10:
            series_forecast = apply_bias_correction(...)
            log_flag = 'bias_adj=True'
        
        # 3. 재계절화
        if mape > 0.20:
            series_forecast = seasonal_recalibration(...)
            log_flag += ',reseason=True'
        
        # 4. Optuna (조건부)
        if mape > 0.20:  # 개선 불충분
            best_params = tune_series(series_id, ...)
            log_flag += ',tuned=True'
        
        results.append({
            'series_id': series_id,
            'mape': mape,
            'adjustments': log_flag
        })
    
    return pd.DataFrame(results)
```

---

## 6️⃣ 운영 가드라인

### 구현: `src/guards.py`

| 항목 | 조건 | 동작 |
|------|------|------|
| 희소도 가드 | 52주 중 0≥80% | Naive 고정 |
| 드리프트 | 평균/분산 3σ 초과 | 재계절화 경고 |
| 실측 누락 | 불완전 연도 | reconcile 보류 |

```python
def check_sparsity_guard(y, threshold=0.8):
    """희소도 체크"""
    zero_ratio = (y == 0).sum() / len(y)
    return zero_ratio >= threshold

def check_drift(y, window=52):
    """드리프트 감지"""
    recent = y[-window:]
    historical = y[:-window]
    
    mean_shift = abs(recent.mean() - historical.mean()) / historical.std()
    var_shift = abs(recent.var() - historical.var()) / historical.var()
    
    return mean_shift > 3 or var_shift > 3
```

---

## 7️⃣ KPI 및 보고

### 목표 지표

| 지표 | 목표 | 개선 기대 |
|------|------|-----------|
| MAPE | ≤ 0.20 | -5~15% 개선 |
| Bias | ≤ 0.05 | 안정화 |
| 재현성 | 100% 동일 입력=동일 결과 | 유지 |

### 산출물

* `artifacts/optuna/optuna_summary.csv`
* `artifacts/metrics/metrics_after_tuning.parquet`
* `reports/roll_{start}_{end}.md` (Before/After 비교)
* `logs/runs_YYYYMMDD.jsonl` (tuned/recalib 플래그)

---

## 🔄 실행 예시

```bash
# 1) 기본 학습
python batch.py train --year 2024

# 2) 롤링 기준선 생성
python batch.py roll --start 2020 --end 2024

# 3) 보정 1차 적용
python batch.py reconcile --year 2024 --kpi-mape 0.20

# 4) 필요 시 튜닝 실행
python tools/run_optuna.py --candidates artifacts/metrics/tuning_candidates.csv

# 5) 최종 보고서
python tools/generate_report.py --year 2024
```

---

## 📁 파일 구조

```
quality-cycles/
├── batch.py                        # CLI 확장 (roll 명령 추가)
├── pipeline_train.py               # 기본 학습 (현재)
├── pipeline_reconcile.py           # 통합 보정 파이프라인
├── src/
│   ├── reconcile.py               # 보정 로직 (bias, seasonal)
│   ├── guards.py                  # 운영 가드라인
│   └── metrics.py                 # MAPE, MASE, Bias 계산
├── tools/
│   ├── validate_baseline.py       # Baseline 진단
│   ├── run_optuna.py             # Optuna 튜닝
│   └── generate_report.py        # 보고서 생성
└── artifacts/
    ├── metrics/                   # 지표 저장
    ├── optuna/                    # 튜닝 결과
    └── models/                    # 모델 아티팩트
```

---

## ✅ 구현 우선순위

1. **Phase 1 (즉시)**: Baseline 검증 + 메트릭 계산
2. **Phase 2 (1주)**: Rolling 백테스트 + Bias/Seasonal 보정
3. **Phase 3 (2주)**: Optuna 통합 + 운영 가드라인
4. **Phase 4 (3주)**: 보고서 자동화 + 모니터링

---

## 💡 핵심 원칙

> **Optuna는 최후의 수단이다. 먼저 경량 보정으로 잡고, 남은 시리즈만 자동 튜닝하라.**
> 
> 이렇게 하면 **운영비용 최소화 + 성능 개선 + 완전 재현성**을 동시에 달성한다.
