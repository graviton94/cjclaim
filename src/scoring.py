# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Dict, Any
from .constants import EWS_WINDOW_RECENT, EWS_WINDOW_BASE

def calculate_prediction_score(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_lower: np.ndarray, y_upper: np.ndarray) -> Dict[str, float]:
    """예측 성능 지표 계산
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        y_lower: 예측 하한
        y_upper: 예측 상한
    
    Returns:
        성능 지표 딕셔너리 (MAPE, R2, 신뢰구간 너비 등)
    """
    # MAPE 계산 (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # R-squared 계산
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # 신뢰구간 너비 (상대적)
    interval_width = np.mean((y_upper - y_lower) / y_pred) * 100
    
    # 종합 신뢰도 점수 계산 (0~100)
    # MAPE가 낮을수록, R2가 높을수록, 신뢰구간이 좁을수록 좋음
    reliability_score = (
        (100 - min(mape, 100)) * 0.4 +  # MAPE 반영 (40%)
        max(min(r2 * 100, 100), 0) * 0.4 +  # R2 반영 (40%)
        (100 - min(interval_width, 100)) * 0.2  # 신뢰구간 너비 반영 (20%)
    )
    
    return {
        "mape": mape,
        "r2": r2,
        "interval_width": interval_width,
        "reliability_score": reliability_score
    }

def early_warning_rule(y: pd.Series) -> dict:
    if len(y) < (EWS_WINDOW_RECENT + EWS_WINDOW_BASE):
        return {"alert": False, "score": 0.0, "recent": np.nan, "baseline": np.nan, "sigma": np.nan}
    recent = float(np.mean(y[-EWS_WINDOW_RECENT:]))
    base_seg = y[-(EWS_WINDOW_RECENT + EWS_WINDOW_BASE):-EWS_WINDOW_RECENT]
    baseline = float(np.mean(base_seg))
    sigma = float(np.std(base_seg, ddof=1))
    alert = recent > (baseline + sigma)
    score = 0.0 if sigma == 0 else (recent - baseline) / sigma
    return {"alert": bool(alert), "score": score, "recent": recent, "baseline": baseline, "sigma": sigma}
