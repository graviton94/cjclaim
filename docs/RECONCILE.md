# Reconcile 보정 시스템

월별 증분학습 후 KPI 미달 시 자동 보정 파이프라인

## 개요

예측 성능이 KPI 목표에 미달할 때 순차적으로 3단계 보정을 실행:

```
Stage 1: Bias Map → Stage 2: Seasonal Recalibration → Stage 3: Optuna Tuning
```

각 단계 후 KPI를 체크하여 목표 달성 시 조기 종료

## KPI 목표

- **MAPE < 20%** (Mean Absolute Percentage Error)
- **|Bias| < 0.05** (절대 편향)

## 보정 단계

### Stage 1: Bias Map

**개념:** 시리즈별 평균 오차를 계산하여 예측값에 단순 보정 적용

**장점:**
- 매우 빠름 (수초 내 완료)
- 계산 비용 최소
- 체계적 편향 제거

**적용 조건:**
- 최소 4주 이상 데이터가 있는 시리즈만 보정
- 보정값 = 주간 평균 오차

**수식:**
```
y_pred_corrected = y_pred + avg_bias
```

### Stage 2: Seasonal Recalibration

**개념:** 최근 2년(104주) 데이터로 계절성 성분 재추정

**장점:**
- 최신 트렌드 반영
- 계절성 패턴 업데이트

**방법:**
1. STL decomposition으로 seasonal 성분 추출
2. 기존 예측의 seasonal과 비교
3. 차이만큼 보정 적용

**현재 상태:** ⚠️ 구현 예정

### Stage 3: Optuna Tuning

**개념:** MAPE 상위 시리즈에 대해 하이퍼파라미터 자동 최적화

**장점:**
- 최적 모델 파라미터 탐색
- 성능 향상 잠재력 최대

**단점:**
- 시간 소요 (timeout 설정 가능)
- 계산 비용 높음

**적용 대상:**
- MAPE 상위 10% 시리즈
- 최소 26주 이상 데이터

**탐색 공간:**
```python
(p, d, q) ∈ [0,3] × [0,2] × [0,3]
(P, D, Q, s) ∈ [0,2] × [0,1] × [0,2] × {52}
```

**현재 상태:** ⚠️ 구현 예정

## 사용법

### CLI

```bash
# 전체 단계 실행
python reconcile_pipeline.py --year 2024 --month 1

# 특정 단계만 실행
python reconcile_pipeline.py --year 2024 --month 1 --stage bias
python reconcile_pipeline.py --year 2024 --month 1 --stage seasonal
python reconcile_pipeline.py --year 2024 --month 1 --stage optuna

# KPI 목표 커스터마이징
python reconcile_pipeline.py --year 2024 --month 1 --kpi-mape 0.15 --kpi-bias 0.03
```

### batch.py 통합

```bash
# 새로운 월별 Reconcile
python batch.py reconcile --month-new 2024-01 --stage-new all

# 특정 단계만
python batch.py reconcile --month-new 2024-01 --stage-new bias
```

### Streamlit UI

```bash
streamlit run app_incremental.py
```

**Tab 3: Reconcile 보정** 사용:
1. 월 선택
2. 보정 단계 선택 (all/bias/seasonal/optuna)
3. "🔧 Reconcile 실행" 버튼 클릭
4. 결과 확인

## 출력 파일

```
artifacts/reconcile/YYYYMM/
├── reconcile_summary_YYYYMM.json          # 전체 요약
├── predict_vs_actual_reconciled_YYYYMM.csv  # 보정된 예측-실측 비교
├── improvement_report_YYYYMM.txt          # 개선 리포트
└── bias_map.csv                           # Stage 1 Bias Map
```

### reconcile_summary 구조

```json
{
  "year": 2024,
  "month": 1,
  "initial_kpi": {
    "MAPE": 0.2534,
    "Bias": 0.0821,
    "MAE": 12.45,
    "RMSE": 18.32,
    "n_records": 1250,
    "n_series": 145
  },
  "stages_run": [
    {
      "stage": "bias_map",
      "improvement": {
        "before_mae": 12.45,
        "after_mae": 10.23,
        "improvement_pct": 17.8,
        "n_series_corrected": 132
      }
    }
  ],
  "final_kpi": {
    "MAPE": 0.1876,
    "Bias": 0.0342,
    "MAE": 10.23,
    "RMSE": 15.67
  },
  "pass": true,
  "timestamp": "2024-01-15T14:32:10"
}
```

## 워크플로우

```
월별 데이터 처리 (process_monthly_data.py)
    ↓
KPI 체크
    ↓
KPI 미달? → YES → Reconcile 실행
    ↓                    ↓
    NO             Stage 1: Bias Map
    ↓                    ↓
  완료              KPI 체크
                         ↓
                   통과? → YES → 완료
                         ↓
                        NO
                         ↓
                   Stage 2: Seasonal
                         ↓
                   KPI 체크
                         ↓
                   통과? → YES → 완료
                         ↓
                        NO
                         ↓
                   Stage 3: Optuna
                         ↓
                   최종 KPI 확인
```

## 성능 개선 예시

**Before Reconcile:**
- MAPE: 25.34%
- |Bias|: 0.0821
- MAE: 12.45

**After Stage 1 (Bias Map):**
- MAPE: 18.76% ✅
- |Bias|: 0.0342 ✅
- MAE: 10.23
- 개선: 17.8%

## 향후 개선

### Stage 2 구현
- [ ] STL decomposition 적용
- [ ] 최근 2년 seasonal 추출
- [ ] Seasonal adjustment 로직

### Stage 3 구현
- [ ] Optuna 연동
- [ ] 탐색 공간 정의
- [ ] Timeout 및 병렬 처리
- [ ] Best params 저장

### 추가 기능
- [ ] 보정 전후 시각화
- [ ] 시리즈별 보정 이력 추적
- [ ] 자동 Reconcile 트리거 (KPI 미달 시)

## 참고

- 보정은 예측값만 수정, 모델은 재학습하지 않음
- 재학습이 필요한 경우 `batch.py retrain` 사용
- Reconcile 결과는 다음 월 처리에 영향 없음 (독립적)
