# Copilot Canvas — Progress snapshot (saved)

## Executive Summary (요약)

이 문서는 CJ 품질클레임 예측엔진의 현재 상태, 품질 게이트, 마일스톤 및 실행 체크리스트를 정리한 배포 로드맵입니다. 현재 최우선 과제는 모델 저장 시 계절성(seasonal)을 올바르게 반영하는 것이며(M1), 이를 해결하면 G3(Seasonal 비중 ≥80%) 달성 가능성이 큽니다. 권장되는 다음 단계는 ① 저장/선택 로직 패치 및 단위검증, ② 샘플(120건) 재학습으로 동작 확인, ③ 전수 재학습 및 AR-root 전수 검사 순입니다.

## Progress update — 2025-11-11

Recent work (artifacts written and checks completed):
- Applied AR-root stability gate and wrote a filtered rerun snapshot and summary CSV (see `artifacts/trained_models_v3_rerun_500_arroot_filtered.json` and `artifacts/arroot_filter_summary.csv`).
- Generated filled feature JSONs (explicit monthly zeros) under `artifacts/features_filled/` (~6029 files).
- Produced `artifacts/zero_months_report.csv` summarizing n_total / n_nonzero / pct_nonzero for filled series.
- Regenerated validation artifacts for seasonal-flag-change candidates and produced per-series PNGs (`artifacts/validation_seasonal_flag_changes.csv`, `artifacts/plots/seasonal_flag_changes/` — 58 series).
- Ran a quick summarizer that reported counts and top AR-root offenders; many unstable seasonal fits were detected and handled by the AR-root gate.

Quick implications and next suggested actions (so next chat can continue immediately):
- Inspect `artifacts/arroot_filter_summary.csv` and a small set of plots under `artifacts/plots/seasonal_flag_changes/` (prioritize highest max_abs_arroot values) to confirm post-gate choices.
- Decide whether to permanently adopt the AR-root gate in the main training flow (recommended) or apply it selectively.
- Persist zeros in canonical feature generation (`data/features/*.json`) so downstream tools and Streamlit always see the same monthly inputs used for training.

## Quick smoke-test (PowerShell)

Run a short smoke test on a small sample to verify the pipeline tools and checks are wired correctly. Two safe options:

- Full rerun sample (will attempt retrain/re-evaluate a small set):

```powershell
python .\tools\rerun_small_batch.py --input artifacts/retry_list_sample10.json --prefer-seasonal --delta-aic 2 --maxiter 200
```

- Read-only check (fast):

```powershell
python .\tools\check_arroot_stability.py --top 20
```

Outputs:
- `artifacts/top20_arroot_summary.csv` and `artifacts/top20_arroot_summary.md` (this session creates a quick top-20 summary if `artifacts/arroot_filter_summary.csv` exists).


---

## 📂 프로젝트 구조 및 기능 요약

**폴더 트리 개요**

```
quality-cycles/
├── data/
│   ├── curated/claims_monthly.parquet       → 2021~2023 백데이터(월 단위)
│   └── features/cycle_features.parquet       → 시계열 피처 엔지니어링 결과
├── src/
│   ├── model_monthly.py                      → SARIMAX/ARIMA/fit_with_retries 구현
│   ├── forecasting.py                        → 예측 로직 및 √변환·역변환
│   ├── schema_validator.py                   → 스키마 및 월 인덱스 검증기
│   └── ews_scoring.py (예정)                 → EWS 스코어 및 급등 감지 로직
├── tools/
│   ├── check_untrained_series.py             → 신규/누락 시리즈 검증
│   ├── check_arroot_stability.py             → AR-root 안정성 검사
│   ├── rerun_small_batch.py                  → 실패/누락 시리즈 재학습
│   └── gen_fit_summary.py (예정)             → fit_summary_log 자동 생성
├── artifacts/
│   ├── trained_models.json                   → 학습된 SARIMAX 모델 결과 저장
│   ├── training_summary.json                 → 전체 학습 통계
│   ├── untrained_series.json                 → 신규/희소 시리즈 (Pending 큐)
│   ├── failed_series.json                    → 학습 실패 시리즈 로그
│   └── stability_report.csv                  → AR-root 안정성 리포트
├── app_incremental.py                        → Streamlit 대시보드 (실측+예측 패널)
├── pipeline_train.py                         → 전체 학습 파이프라인 (SARIMAX 학습)
└── run_monthly_pipeline.ps1 (예정)           → 원클릭 월단위 학습 실행 스크립트

```

## 1) 시스템 개요(요약)

* **데이터**: 월단위 클레임 표준 스키마(6컬럼), 2021~2023 백데이터 + 2024~ 월별 증분 CSV.
* **모델**: SARIMAX(계절 후보), 재시도/안정성 검사(AR-root), √변환, Baseline(희소/저데이터)
* **파이프라인**: 검증→학습→예측→EWS→평가→리포트, Pending/Fail/Trained 큐 분리.
* **운영**: Streamlit(실측12M+예측6M), 월 업로드 원클릭 스크립트, 품질 게이트 G1~G4.

---

## 2) 품질 게이트(Release Gates)

| 코드     | 정의                        | 목표    | 현황(요약)                    |
| ------ | ------------------------- | ----- | ------------------------- |
| **G1** | **비정상적 미저장율**(Pending 제외) | ≤ 2%  | 신규·희소 제외 시 ≈0.8% (달성 가시권) |
| **G2** | **AR-root 안정성**(전수)       | 100%  | Top-표본 OK → 전수 필요         |
| **G3** | **Seasonal 비중**           | ≥ 80% | 현재 저장 스펙상 0% → **최우선 해결** |
| **G4** | **그래프 연속성**               | 통과    | 적용 완료                     |

---

## 3) 마일스톤(최종배포까지 4단계)

### M1. Seasonal 복원 & 저장/선택 경로 교정 (D0~D1)

**목표**: `seasonal_order≠(0,0,0,0)` 저장 정상화, 부분 재학습으로 seasonal_ratio>0 확보.

* 저장 직전 `model_spec.seasonal_order` **assert+로그** 추가, 외부 변수 덮어쓰기 차단.
* **seasonal-first + ΔAIC≤2** 선택규칙 적용(우선 계절 채택).
* 대상: `failed_series(20)` + **랜덤 100** = 120건 샘플 재학습.
* **산출물**: `trained_models_part.json`, `fit_summary_log.json`(order/seasonal/aic/min|root|/n_train).

### M2. Pending/Fail 운영정책 확정 & 전수 재학습 (D1~D2)

**목표**: G1 달성(비정상적 미저장율 ≤2%), Pending/Fail 분리 고도화.

* Pending 허들: `n_train≥24`, `zero_ratio≤0.9`, `Var>0`, `max_zero_run<12`.
* 허들 미달: **Baseline**(mean+trend) 예측 저장(`trainable=false`).
* 전수 재학습: 재시도(optimizers powell→bfgs→lbfgs, maxiter 50→200→400), `<36M`은 ARIMA(1,0,0) 경로.
* **산출물**: `trained_models.json(갱신)`, `untrained_series.json(Pending)`, `failed_series.json(최소화)`.

### M3. 안정성·성능 리포트 자동화 (D2~D3)

**목표**: G2 전수 달성, 성능표준 리포트 완성.

* AR-root 전수 검사: `check_arroot_stability.py --top 6012` → 불안정 0건.
* 평가 표준화: 2024-01~최근월, **ME/MAPE/sME** 롤링 백테스트.
* **산출물**: `stability_report.csv(전수)`, `metrics_report.json`(by series/plant/category & 월별).

### M4. 운영자동화·문서화·릴리즈 (D3~D4)

**목표**: **원클릭 운영**과 문서 완료 → v1.0.0 릴리즈.

* `run_monthly_pipeline.ps1`: CSV→검증→학습/예측→EWS→평가→리포트 자동 실행.
* Streamlit 탭: **EWS Rank**, **Pending 리스트**, **성능 지표 카드** 고정.
* README/운영가이드: Pending/Fail/Trained 정책, 게이트, 장애대응(재시도/대체) 명시.
* **산출물**: 릴리즈 태그 `v1.0.0`, 체인지로그.

---

## 4) 단계별 상세 작업(체크리스트)

### (A) 코드·로직 (필수)

* [ ] **seasonal 저장/선택 경로 확정**: `model_spec={order, seasonal_order, trend='n', enforce_*:True}` 로컬 dict 사용, 저장 직전 `assert` + 로그(`print(f"[SAVE]{series_id} so={seasonal_order}")`).
* [ ] **계절 우선 선택 규칙**: `rank = (is_seasonal==False, selection_loss)`로 정렬 → ΔAIC≤2 내 근소 차이면 계절 유지.
* [ ] **fit_with_retries 일원화**: methods=`['lbfgs','bfgs','powell']`, maxiter=`[50,200,400]`, `mle_retvals['converged']` 체크.
* [ ] **AR-root 검사**: `if hasattr(res,'arroots') and np.any(np.abs(res.arroots)<=1): raise RuntimeError('unstable')`.
* [ ] **Pending 허들 로직**: `n_train>=24` & `zero_ratio<=0.9` & `Var>0` & `max_zero_run<12` → train, else baseline.
* [ ] **Baseline 저장 스펙**: `model_spec.type='baseline_mean_trend'`, `trainable=false`, 예측 6M 생성.
* [ ] **짧은 시리즈 경로**: `<36M`은 `ARIMA(1,0,0)` 경로(재시도 동일), 허들 충족 시 계절 재평가.
* [ ] **에러 처리**: 폴백 금지, 실패는 `failed_series.json`에 원인(optimizer/maxiter/exception) 기록.
* [ ] **유닛 테스트(최소)**: 저장 dict 유실 방지 테스트, ΔAIC 선택 로직 테스트, AR-root 실패 케이스 테스트.

### (B) 도구·배치

* [ ] `tools/check_untrained_series.py` → **Pending/Fail 분리**, 비정상 누락률 계산(총 시리즈 대비 trainable-미저장 비율).
* [ ] `tools/check_arroot_stability.py` → `--top 6012` 지원, CSV 컬럼: `series_id, stable, order, seasonal_order, min_abs_root`.
* [ ] `tools/rerun_small_batch.py` → 입력 리스트(Fail 또는 trainable Pending) 대상으로 재시도, 옵션: `--prefer-seasonal --delta-aic 2 --maxiter 400`.
* [ ] `tools/gen_fit_summary.py` → `fit_summary_log.json` 생성(키: series_id, order, seasonal_order, aic, n_train, min_abs_root, converged).
* [ ] `run_monthly_pipeline.ps1` → 검증→학습→예측→EWS→평가→리포트까지 원클릭, 실패시 재시도 1패스 자동.
* [ ] 로깅 경로 표준화: `artifacts/logs/YYYYMMDD/` 하위에 각 단계 로그 JSONL 저장.

### (C) 리포트·대시보드

* [ ] `metrics_report.json` 생성: 2024-01~현재 월 **ME/MAPE/sME** 롤링; 차원: series/plant/category & 월.
* [ ] Streamlit **성능 KPI 카드**: 최근 3개월 평균 MAPE, 지난달 대비 ΔMAPE, trainable Pending 수량, Fail 수량.
* [ ] Streamlit **EWS Rank 탭**: 급등 트리거(기울기, YoY, Z-score) 점수 및 총합 랭킹, 필터(공장/카테고리/중분류).
* [ ] Streamlit **Pending 탭**: 허들 미달 항목 리스트(사유, 예상 허들 달성 월), Baseline 예측 미니차트.
* [ ] 그래프 표준: 실측 12M(실선)+예측 6M(점선), x축 월 `%Y-%m`, NaN/음수 0 보장.
* [ ] 주간/월간 자동 리포트 파일: `reports/monthly_summary_YYYY-MM.md` 생성(게이트/G1~G4 결과, 상위 EWS Top-20).

### (D) 선택함수·탐색 (To-Do)

* [ ] **선택함수 고도화**: `loss = MAPE + λ*|ME| + α*(1 if nonseasonal else 0)`; 기본 λ=0.2, α=0.5 시험.
* [ ] **Optuna 통합**: 검색공간 (p,q)∈{0,1,2}, seasonal∈{(1,1,1,12),(0,1,1,12)}, λ∈[0,0.5], α∈[0,1]; 목적함수=검증 창 MAPE.
* [ ] **롤링 CV**: 3-fold 월 롤링(학습 24M, 검증 6M ×3), 베스트 하이퍼 선택.
* [ ] **실험 로깅**: `artifacts/optuna_study.db` 및 `study_summary.csv` 저장, 재현 커맨드 기록.
* [ ] **성능 게이트 연동**: 선택함수/Optuna 결과가 기존 대비 ΔMAPE 개선 시에만 채택(예: -3% 이상).
* [ ] `check_untrained_series.py`: Pending/Fail 분리, "비정상적 누락" 계산
* [ ] `gen_fit_summary.py`: `fit_summary_log.json` 자동 생성
* [ ] `rerun_small_batch.py`: 입력 리스트(Fail/Pending 중 trainable) 배치 재시도
* [ ] `run_monthly_pipeline.ps1`: 원클릭 파이프라인(로그/에러 핸들)

<!-- Duplicate section removed: consolidated earlier under (C)/(D) to improve clarity -->
---

## 5) 성공 기준(Definition of Done)

* **G1** 비정상적 미저장율 ≤ 2% (Pending 제외)
* **G2** AR-root 100% 통과(전수)
* **G3** Seasonal 비중 ≥ 80%
* **G4** 시각화 연속성 + 예측 NaN/음수 0
* **성능** 2024-01~최근월 ME≈0, MAPE↓(전월 대비 감소) 리포트 공개
* **운영** 원클릭 파이프라인 정상 동작, README/장애대응 문서화

---

## 6) 일정(요약)

| Day   | 목표                   | 주요 산출물                                         |
| ----- | -------------------- | ---------------------------------------------- |
| D0~D1 | Seasonal 복원·부분 재학습   | trained_models_part.json, fit_summary_log.json |
| D1~D2 | 전수 재학습·Pending 정책 고정 | trained_models.json, untrained/failed 업데이트     |
| D2~D3 | 안정성·성능 전수 리포트        | stability_report.csv, metrics_report.json      |
| D3~D4 | 운영자동화·문서·릴리즈         | run_monthly_pipeline.ps1, README, v1.0.0 태그    |

---

## 7) 위험·대응

| 위험              | 내용               | 대응                                             |
| --------------- | ---------------- | ---------------------------------------------- |
| Seasonal 여전히 0% | 선택·저장 버그 잔존      | 저장 전/후 diff 로그, unit test, 강제 계절 우선 플래그로 일시 고정 |
| 대량 재학습 비용       | 6,012 전수 시간/메모리  | `--max-workers` 제한, 배치 분할, 캐시 활용               |
| 희소 시리즈 성능       | zero-heavy로 변동 큼 | Baseline 유지 + 허들 통과 시 자동 전환                    |

---

### 메모

* “untrained = 에러”가 아니라 **Pending(학습대기)** 로 운영 설계 완료.
* 선택함수 고도화·Optuna는 **To-Do**로 분리(릴리즈 후 점증 개선).
