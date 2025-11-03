# 🚀 Quick Start Guide - Next Steps

학습이 진행 중이라면, 다음 단계를 준비할 수 있습니다.

## 📌 현재 상태 확인

```powershell
# 학습 진행 상황 확인 (터미널에서)
# 학습이 완료되면 artifacts/models/ 에 모델 파일들이 생성됩니다

ls artifacts/models/
```

---

## 🔍 Phase 1: Baseline 검증

학습이 완료되면 즉시 실행하세요.

```powershell
# Baseline 성능 검증
python tools/validate_baseline.py --year 2024

# 출력 확인
cat reports/baseline_report_2024.md
```

**확인할 것:**
- 평균 MAPE, Bias, MASE
- 폴백 모델 사용률
- 잔차 테스트 통과율
- 튜닝 후보 시리즈 수

---

## 📊 Phase 2: Rolling 백테스트 (옵션)

시간이 있다면 과거 데이터로 롤링 백테스트를 실행하여 더 견고한 기준선을 확립하세요.

```powershell
# 2020-2024 롤링 백테스트
python batch.py roll --start 2020 --end 2024

# 결과 확인
cat reports/rolling_backtest_2020_2024.md
```

**주의:** 이 작업은 각 연도마다 학습-예측-평가를 수행하므로 시간이 오래 걸립니다.

---

## 🔧 Phase 3: 경량 보정 적용

실측값이 있는 경우, 보정 파이프라인을 실행하세요.

```powershell
# 2024년 예측 결과 보정
python batch.py reconcile --year 2024

# 메트릭 확인
python -c "import pandas as pd; df=pd.read_parquet('artifacts/metrics/metrics_2024.parquet'); print(df[['series_id','MAPE_original','MAPE_adjusted','bias_adj_applied','seasonal_recal_applied']].head(10))"
```

**적용되는 보정:**
- ✅ Bias Map 보정 (MAPE > 0.10)
- ✅ Seasonal Recalibration (MAPE > 0.20)
- ✅ Changepoint Detection
- ✅ Guards (희소도, 드리프트 체크)

---

## 🎯 Phase 4: Optuna 튜닝 (필요시)

보정 후에도 성능이 낮은 시리즈가 있다면 Optuna로 튜닝하세요.

```powershell
# 튜닝 후보가 있는지 확인
cat artifacts/metrics/tuning_candidates_2024.csv

# Optuna 튜닝 실행 (상위 20개 시리즈만)
python tools/run_optuna.py --candidates artifacts/metrics/tuning_candidates_2024.csv --max-series 20 --timeout 600 --n-trials 40

# 결과 확인
cat reports/optuna_tuning_report.md
```

**주의:** 
- 시리즈당 최대 10분 소요
- 20개 시리즈 = 약 200분 (3.3시간)
- `--max-series` 로 제한 가능

---

## 📈 결과 모니터링

### 생성된 파일들

```
reports/
├── baseline_report_2024.md              # Phase 1 결과
├── rolling_backtest_2020_2024.md       # Phase 2 결과 (옵션)
└── optuna_tuning_report.md             # Phase 4 결과 (필요시)

artifacts/
├── metrics/
│   ├── metrics_baseline_2024.parquet   # Baseline 메트릭
│   ├── tuning_candidates_2024.csv      # 튜닝 후보 목록
│   ├── rolling_metrics_2020_2024.parquet  # Rolling 메트릭 (옵션)
│   └── metrics_2024.parquet            # 보정 후 메트릭
├── adjustments/
│   └── 2024.json                       # 보정 파라미터
└── optuna/
    ├── tuned_params.json               # 튜닝된 파라미터
    └── tuning_results.csv              # 튜닝 결과
```

---

## 🔄 전체 워크플로우 요약

```
1️⃣ 학습 완료 대기
   ↓
2️⃣ Baseline 검증 (validate_baseline.py)
   ↓
   문제 시리즈 식별 → tuning_candidates.csv
   ↓
3️⃣ 보정 적용 (batch.py reconcile)
   ↓
   - Bias 보정
   - Seasonal 재보정
   - Guards 체크
   ↓
4️⃣ 필요시 Optuna 튜닝 (run_optuna.py)
   ↓
   최적 파라미터 발견
   ↓
5️⃣ 재학습 & 재평가
```

---

## 💡 팁

### 빠른 테스트

```powershell
# 단일 시리즈로 먼저 테스트
python tools/validate_baseline.py --year 2024

# 메트릭만 빠르게 확인
python -c "
import pandas as pd
df = pd.read_parquet('artifacts/metrics/metrics_baseline_2024.parquet')
print(f'평균 MAPE: {df.mape.mean():.4f}')
print(f'평균 Bias: {df.bias.mean():.4f}')
print(f'MAPE>0.20 시리즈: {(df.mape>0.20).sum()}개')
"
```

### 진행 상황 로그

```powershell
# 실행 로그 확인
cat logs/runs_*.jsonl | tail -20
```

---

## ❓ 문제 해결

### Import 오류

```powershell
# Python 경로 확인
python -c "import sys; print('\n'.join(sys.path))"

# 프로젝트 루트에서 실행하는지 확인
pwd  # C:\cjclaim\quality-cycles 이어야 함
```

### 데이터 없음

```powershell
# 예측 결과 확인
ls artifacts/forecasts/

# Curated 데이터 확인
ls data/curated/
```

### 패키지 누락

```powershell
# 필요 패키지 재설치
python -m pip install -r requirements.txt
```

---

## 📞 다음 단계

모든 Phase를 완료한 후:

1. **보고서 검토** - reports/ 디렉토리의 모든 .md 파일
2. **메트릭 비교** - Before/After MAPE, Bias 개선율
3. **프로덕션 배포** - 튜닝된 파라미터 적용
4. **모니터링 설정** - 정기적인 성능 체크

---

**현재 학습이 진행 중이므로, 학습 완료 후 Phase 1부터 시작하세요!** 🚀
