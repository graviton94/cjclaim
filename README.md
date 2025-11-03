Quality-cycles — Quick start

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

### Training Pipeline

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
├── app.py                          # Streamlit 웹 앱
├── batch.py                        # CLI 배치 파이프라인
├── pipeline_train.py               # 학습 파이프라인
├── pipeline_forecast.py            # 예측 파이프라인
├── pipeline_reconcile.py           # 보정 파이프라인 (확장됨)
├── roll_backtest.py               # 롤링 백테스트
│
├── src/
│   ├── metrics.py                 # 메트릭 계산 (MAPE, MASE, Bias 등)
│   ├── reconcile.py               # 보정 로직 (Bias, Seasonal, Changepoint)
│   ├── guards.py                  # 운영 가드라인 (희소도, 드리프트 등)
│   ├── forecasting.py             # 예측 모델
│   ├── preprocess.py              # 전처리
│   └── ...
│
├── tools/
│   ├── validate_baseline.py       # Baseline 검증 도구
│   └── run_optuna.py              # Optuna 튜닝 도구
│
├── data/
│   ├── raw/                       # 원본 데이터
│   ├── curated/                   # 전처리된 데이터
│   └── features/                  # 피처 데이터
│
├── artifacts/
│   ├── models/                    # 학습된 모델
│   ├── forecasts/                 # 예측 결과
│   ├── metrics/                   # 성능 메트릭
│   ├── adjustments/               # 보정 파라미터
│   └── optuna/                    # Optuna 튜닝 결과
│
├── reports/                       # 보고서 (Markdown)
├── logs/                          # 실행 로그
└── configs/                       # 설정 파일
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

