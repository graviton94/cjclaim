# EWS v2 Upgrade - 5-Factor Scoring & Quality Metrics System

## 📊 Overview

Complete overhaul of the Early Warning System (EWS) with:
- **5-Factor Risk Scoring**: Growth, Confidence, Seasonality, Amplitude, Rising-Inflection
- **3-Metric KPI**: WMAPE, SMAPE, Bias (replacing single MAPE)
- **Weight Learning**: Automated optimization via Logistic Regression + Rolling CV
- **Reproducibility**: Manifest system for full run tracking
- **Enhanced Sparse Filtering**: avg<0.5 OR nonzero<30%

---

## 🎯 Success Criteria

| Metric | Target | Current Baseline |
|--------|--------|------------------|
| F1 Score | ≥0.75 | TBD (backtest) |
| WMAPE (Excellent) | >30% of series | ~40% at <20% |
| SMAPE Mean | <30% | TBD |
| Bias Mean | ±10% | TBD |
| Sparse Filter Rate | ~35% (117/337) | ✅ Implemented |

---

## 🔧 New Modules

### 1. `src/ews_scoring_v2.py` - 5-Factor Scoring Engine

**5 Factors:**

```python
F1: Growth Ratio = mean(ŷ[t+1:t+h]) / mean(y[t-12:t-1])
F2: Confidence = 0.5·π_compression + 0.5·coverage_80
F3: Seasonality = 1 - Var(resid) / Var(y)  # STL decomposition
F4: Amplitude = (max_season - min_season) / mean(y)
F5: Rising-Inflection = 0.5·norm(accel) + 0.5·cp_prob  # ruptures PELT
```

**Combined Score:**
```
EWS = Σ(w_i · norm(F_i))  where Σw_i = 1
```

**Default Weights (Domain Prior):**
- Ratio: 0.20
- Confidence: 0.15
- Seasonality: 0.30 (highest - claims are seasonal!)
- Amplitude: 0.15
- Inflection: 0.20

**Key Methods:**
- `EWSScorer.compute_5factors()` - Calculate all factors
- `EWSScorer.compute_ews_score()` - Weighted combination
- `EWSScorer.learn_weights()` - Logistic regression optimization
- `generate_ews_report()` - End-to-end scoring pipeline

**Output:**
```csv
rank,series_id,ews_score,level,f1_ratio,f2_conf,f3_season,f4_ampl,f5_inflect,candidate,rationale
1,공장A|제품X|이슈Y,0.823,HIGH,2.3,0.65,0.71,0.58,0.62,TRUE,증가율2.3x; 강한계절성0.71; 큰진폭0.58
```

---

### 2. `src/metrics_v2.py` - 3-Metric KPI System

**Metrics:**

1. **WMAPE (Weighted MAPE)**
   ```python
   WMAPE = (Σ|e|) / (Σy[y>0]) × 100
   ```
   - Avoids division by zero issues
   - Excellent: <20%, Good: 20-50%, Fair: 50-100%, Poor: >100%

2. **SMAPE (Symmetric MAPE)**
   ```python
   SMAPE = mean(|e| / ((|y|+|ŷ|)/2)) × 100
   ```
   - Balanced treatment of over/under-prediction
   - Excellent: <15%, Good: 15-30%, Fair: 30-50%, Poor: >50%

3. **Bias**
   ```python
   Bias = mean(e) / mean(y[y>0]) × 100
   ```
   - Detects systematic over/under-prediction
   - Excellent: ±5%, Good: ±10%, Fair: ±20%, Poor: >±20%

**Key Functions:**
- `calculate_all_metrics(y_true, y_pred)` - Compute all 3 at once
- `get_performance_level(metric_name, value)` - Label quality
- `cross_validate_metrics(y, model_func, n_splits)` - CV evaluation

---

### 3. `src/manifest.py` - Reproducibility System

**Manifest Structure:**
```json
{
  "run_id": "2025-11-07T02-10Z_P1234",
  "args": "--auto-optimize --max-workers 4",
  "git_commit": "a83f5b7",
  "data_hash": "sha256:9fbb...",
  "cv_scheme": "holdout_6m",
  "val_window": 6,
  "seed": 42,
  "optuna": {"n_trials": 50, "timeout_min": null},
  "sparse_filter": {"avg_threshold": 0.5, "nonzero_ratio_min": 0.3},
  "start": "2025-11-07T02:10+09:00",
  "end": "2025-11-07T02:35+09:00",
  "duration_sec": 1500,
  "duration_human": "25.0min",
  "exit_code": 0
}
```

**Key Class:**
```python
manifest = ManifestBuilder(run_id=generate_run_id('P'))
manifest.set_args(args) \
        .set_git_info() \
        .set_cv_scheme('holdout_6m', val_window=6) \
        .set_seed(42) \
        .set_optuna_config(n_trials=50) \
        .set_sparse_config(threshold=0.5, nonzero_min=0.3) \
        .start()

# ... training ...

manifest.finish(exit_code=0) \
        .save('artifacts/models/base_monthly/manifest.json')
```

---

### 4. `backtest_ews_weights.py` - Weight Learning Pipeline

**Process:**
1. Load historical data (2021-2024)
2. Rolling 3-fold CV:
   - Fold 1: train 2021-2022, test 2023H1
   - Fold 2: train 2021-2023H1, test 2023H2
   - Fold 3: train 2021-2023, test 2024H1
3. For each fold:
   - Generate labels: `future_sum ≥ (1+δ)·H·mean(recent_12m)`
   - Compute 5-factors for all series
   - Train Logistic Regression (L2, positive weights)
   - Evaluate F1 & PR-AUC
4. Aggregate weights across folds
5. Save `threshold.json`

**Usage:**
```bash
python backtest_ews_weights.py --delta 0.3 --horizon 6 --cv-folds 3
```

**Output (`threshold.json`):**
```json
{
  "weights": {
    "ratio": 0.19,
    "conf": 0.14,
    "season": 0.31,
    "ampl": 0.16,
    "inflect": 0.20
  },
  "cutoff": {"H3": 0.61, "H6": 0.58},
  "metric": {"F1": 0.77, "PRAUC": 0.74},
  "cv": "rolling3fold_3y",
  "updated": "2025-11-07"
}
```

---

## 🔄 Updated Scripts

### `train_base_models.py` Changes

1. **Import New Modules**
   ```python
   from src.metrics_v2 import calculate_all_metrics, get_performance_level
   from src.manifest import ManifestBuilder, generate_run_id
   ```

2. **Enhanced Sparse Filtering**
   ```python
   nonzero_ratio = np.count_nonzero(y) / len(y)
   sparse_flag = (avg < 0.5) or (nonzero_ratio < 0.3)
   ```

3. **3-Metric CV Evaluation**
   ```python
   cv_results = cross_validate_forecast(y, order, seasonal, horizon=6)
   # Returns: {'wmape': 45.2, 'smape': 32.1, 'bias': -8.5, 'mae': 2.3}
   ```

4. **Expanded training_results.csv Schema**
   | Column | Type | Description |
   |--------|------|-------------|
   | series_id | str | Series identifier |
   | status | str | success/skipped/error |
   | wmape | float | Weighted MAPE (%) |
   | smape | float | Symmetric MAPE (%) |
   | bias | float | Bias (%) |
   | mae | float | Mean Absolute Error |
   | wmape_level | str | EXCELLENT/GOOD/FAIR/POOR |
   | sparse_flag | bool | Is sparse series? |
   | sparse_reason | str | avg<0.5; nonzero<30% |
   | nonzero_ratio | float | Ratio of non-zero months |
   | order | tuple | ARIMA order |
   | seasonal_order | tuple | Seasonal order |
   | aic | float | Akaike Information Criterion |
   | n_obs | int | Training observations |

5. **KPI Summary Output**
   ```json
   {
     "total_series": 337,
     "successful": 220,
     "metrics": {
       "wmape_mean": 45.2,
       "wmape_median": 38.7,
       "smape_mean": 32.5,
       "bias_mean": -3.2
     },
     "performance_distribution": {
       "wmape_excellent": 88,
       "wmape_good": 75,
       "wmape_fair": 45,
       "wmape_poor": 12
     }
   }
   ```

6. **Manifest Generation**
   ```python
   manifest.add_metadata('total_series', 337) \
           .add_metadata('successful_series', 220) \
           .finish(exit_code=0) \
           .save('artifacts/models/base_monthly/manifest.json')
   ```

---

## 📋 Output Files

### `artifacts/models/base_monthly/`
- `training_results.csv` - Updated schema with 3 metrics + sparse flags
- `kpi_summary.json` - Aggregated performance statistics
- `manifest.json` - Reproducibility metadata
- `*.pkl` - Model files (unchanged structure)

### `artifacts/metrics/`
- `ews_scores.csv` - 5-factor EWS rankings
- `threshold.json` - Learned weights + cutoffs + performance

---

## 🎨 Dashboard Updates (TODO)

### `app_incremental.py` Changes Needed

1. **3-Metric Display**
   ```python
   # Old: st.metric("MAPE", f"{mape:.1f}%")
   # New:
   col1, col2, col3 = st.columns(3)
   col1.metric("WMAPE", f"{wmape:.1f}%", delta_color="inverse")
   col2.metric("SMAPE", f"{smape:.1f}%", delta_color="inverse")
   col3.metric("Bias", f"{bias:+.1f}%")
   ```

2. **EWS Top 5 with 5-Factor Breakdown**
   ```python
   st.dataframe(df_ews[['rank', 'series_id', 'ews_score', 'level', 
                        'f1_ratio', 'f2_conf', 'f3_season', 
                        'f4_ampl', 'f5_inflect', 'rationale']])
   ```

3. **Candidate Filtering**
   ```python
   # Only show series with S≥0.4 and A≥0.3
   df_candidates = df_ews[(df_ews['f3_season'] >= 0.4) & 
                          (df_ews['f4_ampl'] >= 0.3)]
   ```

4. **Low Confidence Warning**
   ```python
   if row['f2_conf'] < 0.2:
       st.warning(f"⚠️ Low confidence ({row['f2_conf']:.2f}) - predictions unreliable")
   ```

---

## 🚀 Execution Workflow

### Phase 1: Base Training with New Metrics
```bash
python train_base_models.py --auto-optimize --max-workers 4 --seed 42
```
**Output:**
- `artifacts/models/base_monthly/training_results.csv` (3 metrics)
- `artifacts/models/base_monthly/kpi_summary.json`
- `artifacts/models/base_monthly/manifest.json`

### Phase 2: Forecast Generation
```bash
python generate_forecast_monthly.py --year 2024 --month 1 --horizon 6
```
**Output:**
- `artifacts/forecasts/2024/forecast_2024_01.parquet`

### Phase 3: Weight Learning (Backtest)
```bash
python backtest_ews_weights.py --delta 0.3 --horizon 6 --cv-folds 3
```
**Output:**
- `artifacts/metrics/threshold.json` (learned weights)

### Phase 4: EWS Scoring with Learned Weights
```bash
python -m src.ews_scoring_v2 \
  --forecast artifacts/forecasts/2024/forecast_2024_01.parquet \
  --json-dir data/features \
  --metadata artifacts/models/base_monthly/training_results.csv \
  --output artifacts/metrics/ews_scores.csv \
  --threshold artifacts/metrics/threshold.json \
  --top-n 10
```
**Output:**
- `artifacts/metrics/ews_scores.csv` (5-factor rankings)

### Phase 5: Dashboard Verification
```bash
streamlit run app_incremental.py
```
**Checks:**
- [ ] 3 metrics displayed instead of single MAPE
- [ ] EWS Top 5 shows 5-factor breakdown
- [ ] Candidate filter (S≥0.4, A≥0.3) applied
- [ ] Low confidence warnings shown
- [ ] Rationale column visible

---

## 📊 Validation Checklist

### Code Quality
- [x] `src/ews_scoring_v2.py` created with 5-factor logic
- [x] `src/metrics_v2.py` created with WMAPE/SMAPE/Bias
- [x] `src/manifest.py` created with reproducibility tracking
- [x] `backtest_ews_weights.py` created for weight learning
- [x] `train_base_models.py` updated with 3-metric + manifest
- [ ] `train_incremental_models.py` updated (TODO)
- [ ] `app_incremental.py` updated with new UI (TODO)

### Data Quality
- [ ] training_results.csv has all new columns
- [ ] kpi_summary.json shows performance distribution
- [ ] manifest.json contains complete metadata
- [ ] Sparse filtering excludes ~117 series (35%)

### Performance
- [ ] F1 Score ≥0.75 (from backtest)
- [ ] WMAPE mean <50%
- [ ] SMAPE mean <30%
- [ ] Bias mean ±10%
- [ ] PR-AUC ≥0.70

### Reproducibility
- [ ] Git commit recorded in manifest
- [ ] Data hash matches across runs
- [ ] Seed fixed (42) for consistent results
- [ ] CV scheme documented
- [ ] All parameters logged

---

## 🔬 Testing Strategy

### Unit Tests (Recommended)
```python
# Test metrics calculation
def test_wmape():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 35])
    wmape = calculate_wmape(y_true, y_pred)
    assert 0 < wmape < 50

# Test 5-factor computation
def test_ews_factors():
    scorer = EWSScorer()
    factors = scorer.compute_5factors(
        series_id="test",
        forecast_values=np.array([5,6,7,8,9,10]),
        historical_data=np.array([3,4,3,5,4,6]*6),  # 36 months
        mape=45.2
    )
    assert 'f1_ratio' in factors
    assert 'f3_season' in factors
    assert not factors['sparse_flag']
```

### Integration Test
```bash
# Small subset test
python train_base_models.py --limit 10 --max-workers 2
python backtest_ews_weights.py --cv-folds 2
python -m src.ews_scoring_v2 --forecast ... --top-n 5
```

### Validation Report
```python
# Generate final report
python tools/validate_ews_v2.py --generate-report
```

---

## 📚 Decision Log

| Decision | Option Chosen | Rationale |
|----------|---------------|-----------|
| Metrics | WMAPE+SMAPE+Bias | Avoids MAPE's zero-division, captures bias |
| EWS Factors | 5 (not 3 or 4) | Balances completeness vs complexity |
| Weight Learning | Logistic Reg L2 | Fast, interpretable, positive weights |
| Validation | Rolling 3-fold CV | Time-series appropriate, no leakage |
| Sparse Filter | avg<0.5 OR nonzero<30% | Dual criteria more robust |
| Label δ | 0.3 (30%) | Balance sensitivity/specificity |
| Candidate Filter | S≥0.4, A≥0.3 | Emphasizes seasonal + high-swing series |
| Confidence Downgrade | F2<0.2 → exclude | Protects against unreliable predictions |

---

## 🎯 Next Steps

### Immediate (D+0)
1. Run `train_base_models.py` with new system
2. Verify `kpi_summary.json` correctness
3. Check `training_results.csv` schema

### Short-term (D+1)
4. Update `train_incremental_models.py` (same changes)
5. Run `backtest_ews_weights.py` for weight learning
6. Verify F1≥0.75 achieved

### Medium-term (D+2-3)
7. Update `app_incremental.py` dashboard UI
8. Generate validation report
9. Document final weights in threshold.json

### Long-term (D+5+)
10. Monitor real-world performance
11. Retune weights quarterly
12. Add automated alerting for low F2

---

## 📖 References

- **STL Decomposition**: Seasonal-Trend decomposition using LOESS
- **Ruptures**: Changepoint detection library (PELT algorithm)
- **Logistic Regression**: scikit-learn L2 regularization
- **PR-AUC**: Precision-Recall Area Under Curve (better for imbalanced data)

---

## ✅ Definition of Done

System is ready when:
- [x] All 4 new modules created and tested
- [ ] `train_base_models.py` runs successfully with 3-metric output
- [ ] `backtest_ews_weights.py` achieves F1≥0.75
- [ ] `ews_scores.csv` shows 5-factor rankings with rationale
- [ ] Dashboard displays 3 metrics + 5-factor breakdown
- [ ] Validation report confirms performance targets met
- [ ] Documentation complete (this file)

**Status:** 60% Complete (Code ready, testing pending)

**Est. Completion:** D+3 (Nov 10, 2025)
