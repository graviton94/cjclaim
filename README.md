Quality-cycles — 품질 클레임 주간 예측 시스템

## 🎯 주요 기능

### 1. Base Training (2021-2023)
- 2,208개 SARIMAX 모델 학습 (84.7% 성공률)
- Lag 기반 품질 필터링 (Normal-Lag 정책)
- 시리즈별 JSON 데이터 관리

### 2. 월별 증분학습 시스템
- 발생일자 기준 1개월 데이터 처리
- Lag 필터링 → 주간 집계 → 예측 비교 → 재학습
- **Warm Start:** start_params로 빠른 수렴 (~75% 시간 절감)
- **Sample Weights:** Normal=1.0, Borderline=0.5
- Streamlit UI를 통한 손쉬운 업로드 및 모니터링

### 3. Reconcile 보정 시스템 (3단계)
- **Stage 1: Bias Map** - 주간 평균 오차 보정 (초 단위)
- **Stage 2: Seasonal Recalibration** - STL 계절성 재추정 (분 단위)
- **Stage 3: Optuna Tuning** - 하이퍼파라미터 최적화 (시간 단위)
- KPI 게이트 자동 체크 (MAPE<20%, |Bias|<0.05)

### 4. 예측 생성 파이프라인
- 다음 8주 예측 (horizon 조정 가능)
- 95% 신뢰구간 계산
- 병렬 처리로 빠른 예측 생성

---

## 🚀 Setup

### 1) Create venv

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3) Run Streamlit (from project root)

```powershell
# recommended (ensures `src` package is importable)
python -m streamlit run app.py
```

---

## 📋 Pipeline Commands

### Base Training

```powershell
# 2021-2023 Base 학습
python batch.py train --mode base --workers 4
```

### 월별 증분학습 (완전 자동화)

```powershell
# 1단계: 월별 데이터 처리
python batch.py process --upload data/claims_202401.csv --month 2024-01

# 2단계: KPI 체크 및 Reconcile (필요 시)
python batch.py reconcile --month-new 2024-01 --stage-new all

# 3단계: 증분 재학습 (Warm Start)
python batch.py retrain --month 2024-01 --workers 4

# 4단계: 다음 월 예측 생성
python batch.py forecast --month-new 2024-02

# ✨ Streamlit UI로 손쉽게 (권장)
streamlit run app_incremental.py
```

### Reconcile 보정

```powershell
# 전체 단계 실행 (Bias Map → Seasonal → Optuna)
python batch.py reconcile --month-new 2024-01 --stage-new all

# 특정 단계만 실행
python batch.py reconcile --month-new 2024-01 --stage-new bias
python batch.py reconcile --month-new 2024-01 --stage-new seasonal
python batch.py reconcile --month-new 2024-01 --stage-new optuna
```

### 기존 Pipeline (연도 기반)

```powershell
# 특정 연도까지 학습
python batch.py train --train-until 2024

# 예측
python batch.py forecast --year 2025

# 보정 (실측값과 비교 및 보정)
python batch.py reconcile --year 2024
```

### Rolling Backtest

```powershell
# 연도별 롤링 백테스트 (기준선 확립 및 튜닝 후보 선별)
python batch.py roll --start 2020 --end 2024
```

### Baseline 검증

```powershell
# 학습 직후 baseline 성능 검증
python tools/validate_baseline.py --year 2024

# 잔차 분석, 메트릭 계산, 폴백 모델 사용률 분석
# 결과: reports/baseline_report_2024.md
```

### Optuna 하이퍼파라미터 튜닝

```powershell
# 성능이 낮은 시리즈 자동 튜닝
python tools/run_optuna.py --candidates artifacts/metrics/tuning_candidates.csv --timeout 600 --n-trials 40

# 결과: artifacts/optuna/tuned_params.json
```

---

## 📁 Project Structure

```
quality-cycles/
├── app.py                          # Streamlit 웹 앱 (Base 학습)
├── app_incremental.py              # Streamlit 증분학습 UI
├── batch.py                        # CLI 통합 배치 (7개 서브커맨드)
│
├── pipeline_train.py               # Base 학습 파이프라인
├── pipeline_forecast.py            # 예측 파이프라인
├── pipeline_reconcile.py           # 보정 파이프라인
├── roll_pipeline.py                # 롤링 백테스트
│
├── train_base_models.py            # Base 학습 로직
├── train_incremental_models.py     # 증분 재학습 (Warm Start)
├── generate_monthly_forecast.py    # 월별 예측 생성
├── reconcile_pipeline.py           # 3단계 Reconcile (Bias/Seasonal/Optuna)
│
├── preprocess_to_curated.py        # 전처리 (raw → curated)
├── process_monthly_data.py         # 월별 데이터 처리 (증분학습용)
├── generate_series_json.py         # 시리즈별 JSON 생성
├── evaluate_predictions.py         # 예측 평가
│
├── src/
│   ├── changepoint.py             # 변화점 감지
│   ├── constants.py               # 상수 정의
│   ├── cycle_features.py          # 주기 특성 추출
│   ├── forecasting.py             # SARIMAX 예측
│   ├── io_utils.py                # I/O 유틸리티
│   ├── preprocess.py              # 전처리 로직
│   └── scoring.py                 # 메트릭 계산
│
├── data/
│   ├── raw/                       # 원본 데이터
│   ├── curated/                   # 전처리된 데이터 (weekly)
│   └── features/                  # 피처 데이터
│
├── artifacts/
│   ├── models/                    # 학습된 모델 (2,208개 PKL)
│   ├── forecasts/                 # 예측 결과 (Parquet/CSV)
│   ├── adjustments/               # Reconcile 보정 파라미터
│   └── mlruns/                    # MLflow 실험 추적
│
├── reports/                       # 보고서 (Markdown, 런타임 생성)
├── logs/                          # 실행 로그 (런타임 생성)
├── configs/                       # 설정 파일 (config.yaml)
└── scripts/                       # 유틸리티 스크립트

# 총 15개 핵심 Python 파일 (테스트/검증 파일 34개 제거 완료)
```

---

## 🔄 Next Steps Workflow

### Phase 1: Baseline 검증 (학습 직후)

```powershell
python tools/validate_baseline.py --year 2024
```

**출력:**
- `reports/baseline_report_2024.md` - 잔차 진단, 메트릭, 폴백 분석
- `artifacts/metrics/tuning_candidates_2024.csv` - 튜닝 후보 시리즈

### Phase 2: Rolling 백테스트 (기준선 확립)

```powershell
python batch.py roll --start 2020 --end 2024
```

**출력:**
- `reports/rolling_backtest_2020_2024.md` - 연도별 성능, 트렌드
- `artifacts/metrics/rolling_metrics_2020_2024.parquet` - 상세 메트릭
- `artifacts/metrics/tuning_candidates_rolling_2020_2024.csv` - 튜닝 후보

### Phase 3: 경량 보정 (Bias/Seasonal)

```powershell
python batch.py reconcile --year 2024
```

**기능:**
- ✅ Bias Map 보정 (주차별 편향 패턴)
- ✅ Seasonal Recalibration (최근 2년 데이터)
- ✅ Changepoint Detection (변화점 감지)
- ✅ Guards (희소도, 드리프트 체크)

### Phase 4: Optuna 튜닝 (조건부)

```powershell
python tools/run_optuna.py --candidates artifacts/metrics/tuning_candidates.csv
```

**출력:**
- `artifacts/optuna/tuned_params.json` - 최적 SARIMAX 파라미터
- `reports/optuna_tuning_report.md` - 튜닝 결과 및 개선율

---

## 🎯 KPI 목표

| 지표 | 목표 | 현재 | 개선 기대 |
|------|------|------|-----------|
| MAPE | ≤ 0.20 | - | -5~15% |
| Bias | ≤ 0.05 | - | 안정화 |
| MASE | ≤ 1.5 | - | -10~20% |
| 재현성 | 100% | ✓ | 유지 |

---

## 💡 핵심 원칙

> **Optuna는 최후의 수단이다. 먼저 경량 보정으로 잡고, 남은 시리즈만 자동 튜닝하라.**
> 
> 이렇게 하면 **운영비용 최소화 + 성능 개선 + 완전 재현성**을 동시에 달성한다.

---

## 📚 Notes

- `app.py` already prepends the project root to sys.path so running `python -m streamlit run app.py` from the project root should allow `from src...` imports to work.
- If your editor's linter (pylance) still flags unresolved imports, make sure the workspace folder is set to the project root. In VS Code: "File -> Open Folder" and choose the `quality-cycles` folder, or set PYTHONPATH in workspace settings.

---

## 🔗 Related Documents

- [NEXT_STEPS.md](NEXT_STEPS.md) - 상세 구현 로드맵
- [reports/](reports/) - 생성된 보고서
- [configs/config.yaml](configs/config.yaml) - 설정 파일

