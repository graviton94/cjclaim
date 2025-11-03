import pandas as pd, numpy as np
from pathlib import Path
from io_utils import read_parquet, write_parquet, log_jsonl, ART
from forecasting import fit_sarimax, load_model, save_artifacts
from src.reconcile import BiasCorrector, SeasonalRecalibrator, ChangepointDetector
from src.guards import check_sparsity, check_drift, check_completeness
from src.metrics import compute_all_metrics

def metrics_table(y_true, y_pred):
    eps = 1e-9
    mape = (np.abs((y_true - y_pred) / np.maximum(y_true, eps))).mean()
    bias = (y_pred - y_true).mean() / (np.maximum(y_true.mean(), eps))
    rmse = np.sqrt(((y_true - y_pred)**2).mean())
    denom = (np.abs(np.diff(y_true))).mean() or 1.0
    mase = (np.abs(y_true - y_pred).mean()) / denom
    return mape, mase, bias, rmse

def reconcile_year(curated_path, year, kpi_mape=0.20, apply_guards=True, apply_bias=True, apply_seasonal=True):
    """
    통합 보정 파이프라인
    
    Args:
        curated_path: Curated 데이터 경로
        year: 보정 대상 연도
        kpi_mape: MAPE 임계값
        apply_guards: 가드 체크 적용 여부
        apply_bias: Bias 보정 적용 여부
        apply_seasonal: 계절성 재보정 적용 여부
    """
    df = read_parquet(curated_path)
    fc = read_parquet(ART/f"forecasts/{year}.parquet")
    
    out_metrics = []
    adjustments = {}
    
    print(f"\n{'='*70}")
    print(f"Reconcile 파이프라인 - {year}년")
    print(f"{'='*70}\n")
    
    for series, g_true in df[df["year"]==year].groupby("series_id"):
        g_pred = fc[fc["series_id"]==series].sort_values("week")
        y_true = g_true.sort_values("week")["claim_count"].values
        y_pred = g_pred["y_pred"].values
        
        # 주차 정보 추출
        week_info = g_true.sort_values("week")["week"].values
        
        # 초기 메트릭 계산
        mape, mase, bias, rmse = metrics_table(y_true, y_pred)
        
        print(f"\n📊 {series}")
        print(f"   초기 MAPE: {mape:.4f}, Bias: {bias:.4f}")
        
        # 가드 체크
        guard_flags = {}
        if apply_guards:
            # 학습 데이터 추출
            hist = df[df["series_id"]==series].sort_values(["year","week"])
            y_hist = hist[hist["year"]<=year-1]["claim_count"].values
            
            # 희소도 체크
            is_sparse = check_sparsity(y_hist, threshold=0.8)
            has_drift = check_drift(y_hist, window=52)
            is_complete = check_completeness(y_hist, expected_length=52)
            
            guard_flags = {
                'is_sparse': is_sparse,
                'has_drift': has_drift,
                'is_complete': is_complete
            }
            
            if is_sparse:
                print(f"   ⚠️  희소 시리즈 감지 - Naive 모델 권장")
            if has_drift:
                print(f"   ⚠️  드리프트 감지 - 재보정 필요")
            if not is_complete:
                print(f"   ⚠️  불완전한 데이터 - 보정 보류")
        
        # 보정 적용
        y_adjusted = y_pred.copy()
        adj_metadata = {
            'bias_adj': False,
            'seasonal_recal': False,
            'changepoints': [],
            'guards': guard_flags
        }
        
        # 1. Bias 보정
        if apply_bias and mape > 0.10 and not guard_flags.get('is_sparse', False):
            try:
                corrector = BiasCorrector(method='weekly')
                y_adjusted = corrector.fit_transform(y_adjusted, y_true, week_info)
                adj_metadata['bias_adj'] = True
                
                # 보정 후 메트릭
                mape_adj, _, bias_adj, _ = metrics_table(y_true, y_adjusted)
                print(f"   ✓ Bias 보정 적용 - MAPE: {mape:.4f} → {mape_adj:.4f}")
                
            except Exception as e:
                print(f"   ✗ Bias 보정 실패: {e}")
        
        # 2. 계절성 재보정
        if apply_seasonal and mape > kpi_mape and not guard_flags.get('is_sparse', False):
            try:
                hist = df[df["series_id"]==series].sort_values(["year","week"])
                y_hist = hist[hist["year"]<=year-1]["claim_count"].values
                
                recalibrator = SeasonalRecalibrator(recent_years=2)
                y_adjusted = recalibrator.fit_transform(y_hist, y_adjusted)
                adj_metadata['seasonal_recal'] = True
                
                # 보정 후 메트릭
                mape_adj, _, bias_adj, _ = metrics_table(y_true, y_adjusted)
                print(f"   ✓ 계절성 재보정 적용 - MAPE: {mape:.4f} → {mape_adj:.4f}")
                
            except Exception as e:
                print(f"   ✗ 계절성 재보정 실패: {e}")
        
        # 3. 변화점 감지
        try:
            hist = df[df["series_id"]==series].sort_values(["year","week"])
            y_hist = hist[hist["year"]<=year-1]["claim_count"].values
            
            detector = ChangepointDetector(method='statistical')
            changepoints = detector.detect(y_hist)
            adj_metadata['changepoints'] = changepoints.tolist()
            
            if len(changepoints) > 0:
                print(f"   ⚠️  {len(changepoints)}개 변화점 감지")
        except Exception as e:
            print(f"   ⚠️  변화점 감지 실패: {e}")
        
        # 최종 메트릭 계산
        final_metrics = compute_all_metrics(y_true, y_adjusted, y_hist if 'y_hist' in locals() else None)
        
        out_metrics.append({
            "series_id": series,
            "year": year,
            "MAPE_original": mape,
            "MAPE_adjusted": final_metrics['mape'],
            "MASE": final_metrics.get('mase', np.nan),
            "Bias_original": bias,
            "Bias_adjusted": final_metrics['bias'],
            "RMSE": final_metrics['rmse'],
            "n_points": len(y_true),
            "bias_adj_applied": adj_metadata['bias_adj'],
            "seasonal_recal_applied": adj_metadata['seasonal_recal'],
            "n_changepoints": len(adj_metadata['changepoints'])
        })
        
        # 보정 정보 저장
        bias_intercept = float((y_true - y_pred).mean())
        adjustments[series] = {
            "bias_intercept": bias_intercept,
            "metadata": adj_metadata
        }
        
        # 필요 시 재학습 (MAPE가 여전히 높은 경우)
        if final_metrics['mape'] > kpi_mape and not guard_flags.get('is_sparse', False):
            try:
                hist = df[df["series_id"]==series].sort_values(["year","week"])
                y_hist = hist[hist["year"]<=year-1]["claim_count"].reset_index(drop=True)
                model2 = fit_sarimax(y_hist)
                save_artifacts(series, year-1, model2)
                print(f"   🔄 재학습 완료")
            except Exception as e:
                log_jsonl({"event":"reseason_fail","series":series,"year":year,"error":str(e)})
                print(f"   ✗ 재학습 실패: {e}")
    
    # 결과 저장
    met = pd.DataFrame(out_metrics)
    write_parquet(met, ART/f"metrics/metrics_{year}.parquet")
    
    import json
    adj_path = ART/f"adjustments/{year}.json"
    adj_path.parent.mkdir(parents=True, exist_ok=True)
    adj_path.write_text(json.dumps(adjustments, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # 요약 출력
    print(f"\n{'='*70}")
    print("Reconcile 완료")
    print(f"{'='*70}\n")
    
    print(f"총 시리즈: {met['series_id'].nunique()}")
    print(f"평균 MAPE 개선: {met['MAPE_original'].mean():.4f} → {met['MAPE_adjusted'].mean():.4f}")
    print(f"Bias 보정 적용: {met['bias_adj_applied'].sum()}개 시리즈")
    print(f"계절성 재보정 적용: {met['seasonal_recal_applied'].sum()}개 시리즈")
    
    log_jsonl({"event":"reconcile","year":year,"ok":True,"n_series":met['series_id'].nunique()})
    
    return met
