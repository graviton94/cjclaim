## 요약 - 2025-11-12

오늘 수행한 핵심 작업과 이후 우선순위 작업(단기/중기)을 정리합니다. 이 문서는 지금까지 적용된 패치, 실행 결과, 검증 상태, 그리고 다음에 반드시 수행해야 할 작업들을 한눈에 볼 수 있게 정리한 실행 요약서입니다. GitHub에 커밋/푸시하기 전에 검토해 주세요.

---

## 오늘 한 일 (핵심)

- 아티팩트 복구 및 v3 스냅샷 승격
  - `artifacts/trained_models_v3.json`을 권한 아티팩트로 승격(backup 후 덮어쓰기)하여 downstream이 읽을 수 있는 상태로 복원했습니다.

- 조사 및 진단 도구 추가
  - `tools/analyze_run_results.py` 추가: `trained_models.json` / `failed_series.json` 상태(시리즈 저장수, 실패 메시지 분포, n_total 분포 등)를 요약하여 `artifacts/run_analysis.json`에 저장.
  - 기존 `tools/summary_failed_series.py`, `tools/recompute_training_summary.py`, `tools/inspect_v3_snapshot.py` 등 진단 스크립트를 사용/보완하여 원인 분석을 수행했습니다.

- 파이프라인 버그 수정 (운영 안정성 개선)
  - `src/model_monthly.py`
    - seasonal candidate 처리 버그 수정: 단일 4-튜플 `seasonal_order` 인자를 받을 때 내부에서 정수로 반복되는 문제를 방지하도록 단일 튜플을 리스트로 감싸는 로직 추가. (이슈: "object of type 'int' has no len()")

  - `pipeline_train.py` (여러 패치)
    - 전역 `unique_months` 기반 재색인 분기 제거 및 단일 정책(시리즈별 시작월 → train_until)으로 통일.
    - `train_until_date_fixed`와 `global_start_date`를 도입하여 실행 간 불일치(24/18 vs 37 포인트 혼재) 방지.
    - 최소 학습 길이 판단을 전체 길이(len(y))가 아닌 학습부분(len(y_train), holdout 이후) 기준으로 변경. (기존 문제로 인해 홀드아웃 계산과 섞여 18/24 고정 현상 발생)
    - 비계절 폴백 후보 추가(긴 시리즈에서 계절성 후보 모두 실패 시에도 간단한 AR을 시도).
    - params/yhat/arroots 처리 방식 개선: 복소수 루트/파라미터 처리(실수부 취하거나 magnitude 저장)로 ComplexWarning 제거 및 정보 보존.
    - 빈 스냅샷 덮어쓰기 방지(atomic write + 기존 snapshot non-empty 체크).

- 컴파일/문법 점검
  - `python -m py_compile pipeline_train.py` 결과: COMPILE_RESULT: True (문법적 오류 없음).

---

## 현재 상태 (요약)

- 전체 학습은 현재 진행 중(약 6,012 시리즈 대상). 중간 로그에서 n_total=36, n_train=30 처럼 정상적인 per-series 재색인/홀드아웃이 관찰됨.
- 진단 스크립트(`tools/analyze_run_results.py`) 실행 결과(중간 아티팩트 기준):
  - 저장된 series (baseline_short 포함): 일부는 baseline_short(짧은 학습)로 저장됨.
  - 실패 엔트리 다수 존재(과거에 발생했던 'All seasonal fits failed' 등 메시지 일부는 패치로 해결됨).
- ComplexWarning은 params/yhat/arroots 처리 개선으로 더 이상 출력되지 않음(현재 로그에서 경고 미관찰).

---

## 변경된 파일(주요) 및 목적

- pipeline_train.py
  - 정책 통일(시리즈별 재색인, 학습 컷오프 고정), baseline 조건 개선, AR-root/params 안전 저장, atomic save 및 덮어쓰기 방지.

- src/model_monthly.py
  - seasonal candidate 입력 처리 버그 수정

- tools/analyze_run_results.py (신규)
  - artifacts의 상태 요약(성공/실패/분포) 생성

참고: 이전에 추가/수정된 도구들(`tools/recompute_training_summary.py`, `tools/summary_failed_series.py`, `tools/inspect_v3_snapshot.py`)은 여전히 유효하며 진단에 사용되었습니다.

---

## 당장 해야 할 작업 (단기, 우선순위)

1. 전체 파이프라인 실행 완료 후 결과 분석 (필수)
   - 실행이 완료되면 아래을 실행해서 아티팩트의 일관성/실패 원인을 확인합니다:
     - `python tools/analyze_run_results.py`
     - `python tools/diag_snapshot.py` (또는 순서대로: `normalize_snapshot.py` → `gen_fit_summary_from_snapshot.py` → `check_arroot_stability.py`)

2. 실패 상위 원인 우선 확인 (우선순위 높은 메시지부터)
   - 예: "All seasonal fits failed" 또는 "y_true and y_pred must have the same shape" 등 남아있는 메시지의 샘플 ID를 추적하여 데이터/전처리 이슈인지 모델링 이슈인지 판단합니다.

3. 샘플 검증(권장, 빠른 피드백)
   - 소규모(예: 100 시리즈)로 빠른 재실행을 해 패치 영향 확인. (pipeline_train.py를 임시로 `series_list = series_list[:100]`로 제한하거나 별도 샘플 모드 추가)

4. 푸시 전 로컬 점검
   - 모든 변경 파일에 대해 `python -m py_compile` 실행 및 간단한 smoke test(샘플 실행)를 권장.

---

## 중기·장기 체크리스트 (권장 개선/리팩토링)

1. 후보 선택 정책(정책적 결정)
   - 현재는 loss 우선 후 AIC 기반 계절 승격 정책(`prefer_seasonal_by_aic`)을 사용합니다. 조직의 목표(G3)에 따라 AIC 우선 혹은 계절성 가중치 정책 도입을 검토.

2. 단위 테스트 및 통합 테스트 추가
   - `src/model_monthly.py`와 `pipeline_train.py`의 핵심 함수(재색인, holdout 계산, candidate fitting 흐름)에 대한 유닛 테스트 작성.

3. 운영 모니터링/알림
   - 장기적으로 실패 비율(예: failed_series 비율), 평균 AIC 분포, min_abs_arroot 분포를 모니터링하고 임계치 초과 시 알림 발송.

4. 스냅샷/병합 정책 강화
   - `normalize_snapshot.py`를 파이프라인의 마지막 단계로 강제하여 스키마 일관성 보장.
   - 빈 스냅샷 덮어쓰기 방어를 계속 유지하되, 자동 복구/프로모션 절차(예: 성공한 마지막 snapshot으로 롤백) 문서화.

5. 성능 최적화
   - SARIMAX fitting 병목(특히 많은 시리즈에서)이 남아 있으므로, 후보 수 축소, 리트라이 전략 개선, 병렬화(프로세스 풀) 도입을 검토.

---

## GitHub에 커밋/푸시 순서 (권장)

1. 변경 내용 스테이징/커밋

```powershell
Set-Location 'c:/cjclaim/quality-cycles'
git checkout -b fix/pipeline-index-arroots-20251112
git add pipeline_train.py src/model_monthly.py tools/analyze_run_results.py docs/EXEC_SUMMARY_2025-11-12.md
git commit -m "train: unify per-series indexing, robust arroots/params handling; add run analysis tool; exec summary"
```

2. 푸시 및 PR 생성

```powershell
git push origin HEAD
# 그런 다음 GitHub에서 PR 생성(타겟 브랜치: main 또는 team-branch)
```

3. PR 템플릿에 포함할 내용
   - 수정 목적 요약, 재현된 문제(증상), 적용한 수정(짧게), 실행/검증 결과(컴파일 성공, 샘플 실행 결과 요약), 리스크/후속 작업 항목

---

## 부가: 빠르게 실행할 때 쓰는 명령 요약

```powershell
# 전체 파이프라인 실행
Set-Location 'c:/cjclaim/quality-cycles'
python pipeline_train.py

# (실행 후) 진단 요약
python tools/analyze_run_results.py
python tools/diag_snapshot.py

# 문법 검사
python -m py_compile pipeline_train.py src/model_monthly.py
```

---

파일 위치: `docs/EXEC_SUMMARY_2025-11-12.md`

필요하시면 제가 그대로 커밋/푸시(브랜치 생성, PR 텍스트 자동 작성)까지 진행해 드립니다. 어떤 방식으로 진행할까요? (제가 푸시해도 될지, 또는 먼저 로컬에서 추가 확인을 하실지 선택해 주세요.)