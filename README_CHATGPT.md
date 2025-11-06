# 품질 클레임 주간 예측 시스템 - ChatGPT 분석용 README

> **⚠️ 이 문서는 ChatGPT가 코드베이스를 분석하기 위한 구조적 가이드입니다.**  
> JSON/PKL 파일은 데이터 구조만 명시하며, 실제 내용 분석은 불필요합니다.  
> **현재 코드 구조상 충돌이나 수정이 급한 사항에 집중하여 작성되었습니다.**

---

## 🎯 프로젝트 핵심 요약

**목적**: CJ 제일제당 품질 클레임 데이터 → 주간 발생 건수 예측 (SARIMAX 시계열 분석)  
**규모**: 967,568건 데이터 → 2,608개 시리즈 → 2,208개 학습 모델 (84.7% 성공률)  
**KPI**: MAPE ≤ 0.20 (20%), Bias ≤ 0.05 (5%)

**주요 특징**:
- 월별 증분학습 (Incremental Learning with Warm Start)
- 3단계 Reconcile 보정 (Bias Map → Seasonal → Optuna)
- 2,608개 시리즈별 개별 모델 (공장_제품_클레임유형)

---

## 📁 폴더 구조 및 주요 역할

```
quality-cycles/
│
├── 📌 CLI & WEB APP (진입점)
│   ├── batch.py                      # CLI with 7 subcommands
│   ├── app.py                        # Streamlit GUI (기본)
│   └── app_incremental.py            # Streamlit GUI (증분학습 전용)
│
├── 🔄 PIPELINES (핵심 워크플로우)
│   ├── pipeline_train.py             # 초기 학습 (2021-2023 base)
│   ├── pipeline_forecast.py          # 예측 실행 (8주)
│   ├── pipeline_reconcile.py         # 3단계 보정
│   └── roll_pipeline.py              # 예측+Reconcile 통합
│
├── 🧠 CORE LOGIC (비즈니스 로직)
│   ├── train_base_models.py          # SARIMAX 초기 학습
│   ├── train_incremental_models.py   # 증분 재학습 (Warm Start)
│   ├── generate_monthly_forecast.py  # 월별 예측 생성
│   └── reconcile_pipeline.py         # Reconcile 실행 로직
│
├── 📊 DATA PROCESSING (데이터 처리)
│   ├── preprocess_to_curated.py      # CSV → Parquet (주간 집계)
│   ├── process_monthly_data.py       # 월별 증분학습 파이프라인
│   ├── generate_series_json.py       # 시리즈별 JSON 생성
│   └── evaluate_predictions.py       # 예측 성능 평가
│
├── 🛠️ SRC (공유 유틸리티 모듈)
│   ├── changepoint.py                # Ruptures 기반 변화점 검출
│   ├── constants.py                  # SplitConfig dataclass, 상수
│   ├── cycle_features.py             # 주기성 feature 추출
│   ├── forecasting.py                # SARIMAX 예측/재학습 로직
│   ├── io_utils.py                   # I/O 헬퍼
│   ├── preprocess.py                 # 전처리 함수
│   ├── reconcile.py                  # BiasCorrector, SeasonalRecalibrator
│   └── scoring.py                    # MAPE, Bias, MASE 계산
│
├── 🔧 TOOLS (분석 유틸리티)
│   ├── lag_analyzer.py               # Lag 이상치 분석 (Normal/Borderline/Extreme)
│   ├── compare_forecast_actual.py    # 예측-실측 비교
│   ├── filter_monthly_data.py        # 월별 Lag 필터링
│   ├── run_optuna.py                 # 하이퍼파라미터 자동 튜닝
│   └── validate_baseline.py          # 베이스라인 검증
│
├── 📦 DATA (데이터 파일)
│   ├── raw/                          # 원시 CSV (claims(2020_2024).csv)
│   ├── curated/                      # 전처리 Parquet (주간 집계)
│   └── features/
│       ├── series_2021_2023/         # 2,608개 시리즈 JSON 파일 ⚠️
│       └── cycle_features.parquet    # 주기성 feature
│
├── 🗄️ ARTIFACTS (모델 & 결과물)
│   ├── models/
│   │   └── base_2021_2023/           # 2,208개 PKL 모델 파일 ⚠️
│   ├── forecasts/                    # 예측 결과 (Parquet)
│   ├── adjustments/                  # Reconcile 보정 파라미터
│   ├── incremental/YYYYMM/           # 월별 증분학습 결과
│   ├── reconcile/YYYYMM/             # 월별 Reconcile 결과
│   └── mlruns/                       # MLflow 실험 추적
│
├── ⚙️ CONFIGS
│   └── config.yaml                   # 경로, KPI, 설정 (YAML)
│
├── 📖 DOCS
│   ├── INCREMENTAL_LEARNING.md       # 증분학습 워크플로우 상세
│   └── RECONCILE.md                  # Reconcile 3단계 설명
│
└── 📝 LOGS
    └── *.log                         # 실행 로그
```

---

## 🔗 주요 코드 파일 간 상관관계

### **1. CLI 계층 (batch.py)**
```
batch.py (7 subcommands)
  ├─ train      → train_base_models.py
  ├─ forecast   → pipeline_forecast.py
  ├─ reconcile  → pipeline_reconcile.py
  ├─ roll       → roll_pipeline.py
  ├─ process    → process_monthly_data.py
  └─ retrain    → train_incremental_models.py
```

### **2. 데이터 흐름**
```
[Raw CSV]
    ↓
preprocess_to_curated.py → [Curated Parquet]
    ↓
generate_series_json.py → [Series JSON 2,608개]
    ↓
train_base_models.py → [PKL Models 2,208개]
    ↓
pipeline_forecast.py → [Forecast Parquet]
    ↓
pipeline_reconcile.py → [Adjusted Forecast]
```

### **3. 증분학습 흐름**
```
[월별 업로드 CSV]
    ↓
process_monthly_data.py
    ↓
├─ tools/lag_analyzer.py (Lag 필터링)
├─ tools/compare_forecast_actual.py (예측-실측 비교)
└─ train_incremental_models.py (모델 재학습 with Warm Start)
    ↓
[업데이트된 PKL Models]
```

### **4. Reconcile 3단계**
```
pipeline_reconcile.py
    ↓
├─ Stage 1: src/reconcile.py (BiasCorrector - 주간 평균 오차 보정)
├─ Stage 2: src/reconcile.py (SeasonalRecalibrator - STL 계절성 재추정)
└─ Stage 3: tools/run_optuna.py (하이퍼파라미터 튜닝)
```

---

## 📋 데이터 구조 (JSON/PKL 메타데이터)

### **⚠️ 시리즈 JSON** (`data/features/series_2021_2023/*.json`)
**구조**: 각 파일은 단일 시리즈의 시계열 데이터
```json
{
  "series_id": "진천BC_백미_미생물",
  "data": [
    {"year": 2021, "week": 1, "y": 0, "lag_label": "normal"},
    {"year": 2021, "week": 2, "y": 2, "lag_label": "normal"}
  ]
}
```
**필드**:
- `year`, `week`: ISO 주차
- `y`: 클레임 발생 건수
- `lag_label`: Lag 이상치 분류 (normal/borderline/extreme)

**파일 수**: 2,608개 (각 시리즈당 1개)

### **⚠️ 모델 PKL** (`artifacts/models/base_2021_2023/*.pkl`)
**구조**: statsmodels SARIMAX 학습 결과 객체
```python
# pickle 직렬화된 SARIMAXResultsWrapper
model = pickle.load(open("model.pkl", "rb"))
# 예측: model.forecast(steps=26)
# 재학습: model.append(new_data, refit=True, start_params=model.params)
```
**파일 수**: 2,208개 (학습 성공한 시리즈)

### **Forecast Parquet** (`artifacts/forecasts/`)
**구조**: 예측 결과 테이블
```python
# 컬럼: series_id, forecast_week, forecast_date, predicted_value, lower_bound, upper_bound
df = pd.read_parquet("forecast_2024_01.parquet")
```

### **Config YAML** (`configs/config.yaml`)
```yaml
paths:
  input: "data/raw/claims(2020_2024).csv"
  output: "data/curated/"

kpis:
  mape: 0.20
  bias: 0.05

engine:
  max_workers: 4
  timeout_tuning_min: 10
```

---

## ⚠️ 현재 코드 구조 상 충돌 및 수정 급한 사항

### **🔴 CRITICAL 1: Reconcile 순서 의존성 미보장**
**파일**: `pipeline_reconcile.py`  
**문제**:
- CLI에서 `--stage-new` 옵션으로 개별 단계 실행 가능
- 하지만 Stage 1 → 2 → 3 순서 보장이 필요

**현재 구조**:
```python
# batch.py에서 개별 실행 가능
python batch.py reconcile --stage-new bias      # Stage 1만
python batch.py reconcile --stage-new seasonal  # Stage 2만 (위험)
python batch.py reconcile --stage-new optuna    # Stage 3만 (위험)
```

**해결 방법**:
```python
# pipeline_reconcile.py에 의존성 체크 추가
def run_stage2_seasonal(forecast_df, month_key):
    bias_map_file = f"artifacts/adjustments/bias/{month_key}_bias_map.parquet"
    if not os.path.exists(bias_map_file):
        raise ValueError("Stage 1 (Bias Map)을 먼저 실행하세요!")
    # ... 이후 로직
```

**영향 범위**: `pipeline_reconcile.py`, `batch.py`

---

### **🟡 WARNING 1: Lag 필터링 정책 코드 중복**
**파일**: `tools/lag_analyzer.py`, `tools/filter_monthly_data.py`, `process_monthly_data.py`  
**문제**:
- Lag 분류 로직이 3곳에서 중복 구현
- Normal (weight=1.0) + Borderline (weight=0.5) 정책이 하드코딩

**현재 상황**:
```python
# tools/lag_analyzer.py (기준 구현)
df['lag_class'] = 'extreme'
# ... 복잡한 if-else 로직 (100+ lines)

# tools/filter_monthly_data.py (복사본)
# 동일 로직 복사됨

# process_monthly_data.py (또 다른 복사본)
# 일부 수정된 버전
```

**해결 방법**:
```python
# src/lag_classifier.py (새 파일 생성)
class LagClassifier:
    def __init__(self, lag_stats_path):
        self.lag_stats = pd.read_csv(lag_stats_path)
    
    def classify(self, df):
        # 통합된 분류 로직
        return df_with_lag_class
```

**영향 범위**: 3개 파일 모두 수정 필요

---

### **🟡 WARNING 2: MLflow 실험 ID 충돌 가능성**
**파일**: `train_base_models.py`, `tools/run_optuna.py`, `train_incremental_models.py`  
**문제**: 동일 실험명으로 여러 run 생성 시 ID 충돌 가능

**현재**:
```python
# train_base_models.py
mlflow.set_experiment("quality_cycles")  # 고정된 이름

# tools/run_optuna.py
mlflow.set_experiment("quality_cycles")  # 동일 이름 (충돌 위험)
```

**해결 방법**:
```python
# 각 파일에서 고유 실험명 사용
experiment_name = f"quality_cycles_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
mlflow.set_experiment(experiment_name)
```

---

### **🟢 INFO 1: Warm Start 파라미터 전달 미비**
**파일**: `train_incremental_models.py`, `src/forecasting.py`  
**상황**: 증분학습 시 `start_params` 전달이 일부 경로에서 누락

**현재**:
```python
# train_incremental_models.py
# start_params를 추출하지만 일부 경우에만 사용
```

**권장사항**: 모든 재학습 경로에서 `start_params` 사용 확인

---

### **🟢 INFO 2: 경로 하드코딩 (일부 파일)**
**문제**: 일부 스크립트에서 절대 경로 하드코딩
```python
# ❌ 나쁜 예
csv_path = "C:/cjclaim/quality-cycles/data/raw/claims.csv"

# ✅ 좋은 예
from pathlib import Path
base_dir = Path(__file__).parent
csv_path = base_dir / "data" / "raw" / "claims.csv"
```

**영향 파일**: `scripts/build_dataset.py`, 일부 tools/

---

### **🟢 INFO 3: 모델 파일 Git 관리**
**상황**: 2,208개 PKL 파일 (2.4 MiB) Git에 업로드됨  
**권장사항**: `.gitignore`에 `artifacts/models/*.pkl` 추가 또는 Git LFS 사용

---

## 🚀 실행 흐름 (ChatGPT가 이해해야 할 순서)

### **Step 1: 초기 학습 (One-time)**
```bash
# 1. 전처리
python preprocess_to_curated.py --input data/raw/claims(2020_2024).csv

# 2. 시리즈 JSON 생성
python generate_series_json.py

# 3. 베이스 모델 학습 (2021-2023)
python batch.py train --mode base --workers 4
```

### **Step 2: 예측 (매월)**
```bash
# 예측 실행 (8주)
python batch.py forecast --month-new 2024-02

# Reconcile 보정 (KPI 미달 시)
python batch.py reconcile --month-new 2024-01 --stage-new all
```

### **Step 3: 증분학습 (매월)**
```bash
# 월별 업로드 → 학습 → 모델 업데이트
python batch.py process --upload data/uploaded/claims_2024_01.csv --month 2024-01
python batch.py retrain --month 2024-01 --workers 4
```

---

## 🔍 주요 의존성 (requirements.txt)

```
pandas==2.0.0           # 데이터 처리
numpy==1.24.0
statsmodels==0.14.0     # SARIMAX 모델
streamlit==1.24.0       # GUI
ruptures==1.1.8         # 변화점 검출
optuna==3.2.0           # 하이퍼파라미터 튜닝
mlflow==2.4.0           # 실험 추적
PyYAML==6.0             # config.yaml 읽기
```

---

## 📝 ChatGPT 분석 가이드

**분석 시 집중할 포인트**:
1. ✅ **데이터 흐름**: Raw CSV → Curated → JSON → PKL → Forecast
2. ✅ **모듈 의존성**: batch.py → pipelines → src/
3. ⚠️ **순서 의존성**: Reconcile 3단계 순서 보장 필요
4. ⚠️ **코드 중복**: Lag 분류 로직 통합 필요
5. ⚠️ **MLflow 충돌**: 실험 ID 고유화 필요

**무시해도 되는 부분**:
- JSON/PKL 파일의 실제 데이터 내용 (구조만 이해)
- MLflow UI 사용법
- Streamlit GUI 디자인

---

## 📚 참고 문서

- [docs/INCREMENTAL_LEARNING.md](docs/INCREMENTAL_LEARNING.md): 증분학습 상세
- [docs/RECONCILE.md](docs/RECONCILE.md): Reconcile 3단계 설명
- [README.md](README.md): 사용자용 기본 README

**작성 일자**: 2024-01-31  
**최종 업데이트**: 2024-01-31  
**버전**: v1.0-chatgpt-optimized
