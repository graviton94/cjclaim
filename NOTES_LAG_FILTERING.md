# Lag Filtering 방법 비교 및 정리

## 📌 현황

현재 프로젝트에 **2가지 Lag 필터링 방법**이 존재합니다:

### 1. tools/lag_analyzer.py (기존, **권장 사용**)

**방법:** μ+σ (평균 + 표준편차) 기준 분류 + 잘못 접수 케이스 제외

**분류 기준:**
```python
Invalid:    lag < 0 (제조일자 > 발생일자) → 제외 (잘못 접수된 케이스)
Normal:     0 ≤ lag ≤ μ + 1σ              → weight = 1.0 (학습 우선)
Borderline: μ+1σ < lag ≤ μ+2σ             → weight = 0.5 (보조 학습)
Extreme:    lag > μ + 2σ                  → 제외 (노이즈)
```

**특징:**
- ✅ **이미 프로젝트에서 사용 중** (README, SYSTEM_SUMMARY 문서화됨)
- ✅ **잘못 접수 케이스 자동 제외** (제조일자가 발생일자보다 미래인 경우)
- ✅ 영구 기준 파일: `artifacts/metrics/lag_stats_from_raw.csv`
- ✅ 제품범주2별 통계 (392개 카테고리)
- ✅ 약 95% 데이터 보존 (Normal+Borderline, Invalid 제외)
- ✅ Sample weights 적용 가능 (statsmodels SARIMAX fit)

**사용법:**
```powershell
# Lag 통계 생성 (최초 1회)
python tools/lag_analyzer.py `
  --input data/raw/claims_merged.csv `
  --output artifacts/metrics/lag_stats_from_raw.csv

# 필터링 적용
python tools/lag_analyzer.py `
  --input data/curated/claims_2024_01.csv `
  --ref artifacts/metrics/lag_stats_from_raw.csv `
  --policy-out data/curated/claims_2024_01_filtered.csv
```

**출력:**
- Normal/Borderline/Extreme/Invalid 라벨링
- lag_category 컬럼 추가
- sample_weight 컬럼 (1.0 / 0.5 / 제외)
- ⚠️ Invalid (음수 lag) 자동 제외 로그 출력

---

### 2. filter_normal_lag.py (신규, **현재 미사용**)

**방법:** IQR (Interquartile Range) 또는 백분위수 기준

**분류 기준 (예시):**
```python
# IQR 방식
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Normal: lag ≤ Q3 + 1.5·IQR

# 백분위수 방식
Normal: lag ≤ 95th percentile
```

**특징:**
- ❌ 아직 문서화되지 않음
- ❌ 기존 파이프라인과 통합되지 않음
- ✅ 극단치 제거에 강건 (outlier-robust)
- ✅ 비대칭 분포에 유리

**현재 상태:**
- Phase 22에서 생성됨
- tools/lag_analyzer.py 발견 전 작성
- 중복 기능으로 판단

---

## 🔍 두 방법 비교

| 항목 | μ+σ 방식 (lag_analyzer.py) | IQR 방식 (filter_normal_lag.py) |
|------|---------------------------|--------------------------------|
| **분류 기준** | 평균 + 표준편차 | 사분위수 범위 |
| **극단치 민감도** | 높음 (outlier가 μ,σ 영향) | 낮음 (median 기반) |
| **데이터 보존율** | ~95% (μ+2σ 내) | 설정에 따라 다름 (~75-95%) |
| **Sample Weights** | ✅ 3단계 (1.0/0.5/0.0) | ❓ 구현 필요 |
| **영구 기준 파일** | ✅ lag_stats_from_raw.csv | ❌ 없음 |
| **문서화** | ✅ README, SYSTEM_SUMMARY | ❌ 없음 |
| **파이프라인 통합** | ✅ 완료 | ❌ 미완 |
| **적합한 분포** | 정규분포 가까운 경우 | 비대칭/긴 꼬리 분포 |

---

## 💡 권장 사항

### 🎯 Option 1: lag_analyzer.py 단독 사용 (현 상태 유지)

**근거:**
- 이미 프로젝트 전체에 통합됨
- 문서화 완료 (README, SYSTEM_SUMMARY)
- 영구 기준 파일 (lag_stats_from_raw.csv) 존재
- Sample weights 적용 가능

**액션:**
- filter_normal_lag.py 삭제 또는 `tools/archive/` 폴더로 이동
- 불필요한 중복 제거

---

### 🔬 Option 2: 비교 실험 후 결정 (데이터 주도)

**실험 설계:**
1. **동일 데이터셋**에 두 방법 적용
2. **학습 성능** 비교:
   - WMAPE/SMAPE/Bias
   - F1 Score (EWS)
   - 학습 시간
3. **데이터 보존율** 비교
4. **극단치 처리** 효과 비교

**코드 예시:**
```powershell
# μ+σ 방식
python tools/lag_analyzer.py --input data/raw/claims_merged.csv --output data/method1_filtered.csv

# IQR 방식
python filter_normal_lag.py --input data/raw/claims_merged.csv --output data/method2_filtered.csv

# 학습 및 비교
python train_base_models.py --input data/method1_filtered.csv --output models/method1/
python train_base_models.py --input data/method2_filtered.csv --output models/method2/

# 성능 비교
python tools/compare_methods.py --method1 models/method1/ --method2 models/method2/
```

**결정 기준:**
- WMAPE Excellent >30% 달성한 방법
- F1 Score ≥0.75 달성한 방법
- 계산 효율성 (처리 시간, 데이터 보존율)

---

### 🔀 Option 3: 하이브리드 (두 방법 통합)

**아이디어:** IQR로 극단치 제거 → μ+σ로 세분화

```python
Step 1: IQR로 1차 필터링 (극단치 제거)
  → lag > Q3 + 1.5·IQR 제외

Step 2: 남은 데이터에 μ+σ 적용 (3단계 분류)
  → Normal/Borderline 구분 (sample weights)
```

**장점:**
- IQR의 강건성 + μ+σ의 세밀한 분류
- Sample weights 적용 가능

**단점:**
- 복잡도 증가
- 검증 필요 (과도한 필터링 위험)

---

## 📝 현재 결정 사항

### ✅ Phase 1 (Fresh Start) - lag_analyzer.py 사용

**이유:**
- 이미 검증된 방법 (구버전에서 2,208개 모델 학습 성공)
- 문서화 및 파이프라인 통합 완료
- 빠른 시작 우선 (데이터 업로드 대기 중)

**액션:**
- filter_normal_lag.py는 보류 (삭제하지 않고 보존)
- Fresh start 학습 완료 후 성능 확인
- 필요 시 Phase 2에서 비교 실험

---

### 🔮 Phase 2 (성능 최적화) - 비교 실험 고려

**조건:**
- Base training (2021-2023) 완료
- KPI 목표 달성 여부 확인 (WMAPE Excellent >30%, F1≥0.75)

**실험 시나리오:**
- 목표 미달 시: IQR 방식 실험 (극단치 영향 제거 효과 검증)
- 목표 달성 시: 현 상태 유지

---

## 📂 파일 정리

### 현재 구조
```
quality-cycles/
├── tools/
│   └── lag_analyzer.py              # ⭐ 사용 중 (μ+σ)
├── filter_normal_lag.py             # ❓ 보류 (IQR)
└── NOTES_LAG_FILTERING.md           # 본 문서
```

### 제안 (Option 1 선택 시)
```
quality-cycles/
├── tools/
│   ├── lag_analyzer.py              # ⭐ Primary method
│   └── archive/
│       └── filter_normal_lag.py     # 보존 (미래 참조용)
└── docs/
    └── LAG_FILTERING_COMPARISON.md  # 비교 문서 (실험 시)
```

---

## 🚀 다음 단계

1. **사용자 확인**: 어떤 방법 사용할지 결정
   - Option 1: lag_analyzer.py 단독 (권장)
   - Option 2: 비교 실험 후 결정
   - Option 3: 하이브리드 통합

2. **파일 정리**: 결정에 따라
   - filter_normal_lag.py 이동/삭제
   - 문서 업데이트

3. **Fresh Start 진행**:
   - 사용자가 2021-2023 데이터 업로드
   - 선택된 방법으로 Lag 필터링
   - Base training 실행

---

**작성일:** 2025-01-13  
**Status:** Decision Pending - 사용자 확인 필요 ⏳
