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
        성능 지표 딕셔너리 (MAE, RMSE, 신뢰구간 적정성 등)
    """
    # Ensure counts cannot be negative: clip inputs to >= 0 for all calculations
    y_true = np.asarray(y_true, dtype=float)
    y_true = np.clip(y_true, 0.0, None)

    # 기본 통계량 (based on non-negative true values)
    y_mean = np.mean(y_true)
    y_std = np.std(y_true) if len(y_true) > 1 else 1.0
    # 안전한 denominators
    denom_mae = y_mean + y_std if (y_mean + y_std) > 0 else 1.0
    denom_rmse = y_mean + 2 * y_std if (y_mean + 2 * y_std) > 0 else 1.0

    # 역사적 범위 기반 클리핑 (극단치 방지)
    historical_max = np.max(y_true) * 1.5 if len(y_true) > 0 else np.max([1.0])
    historical_min = max(0.0, np.min(y_true) * 0.5) if len(y_true) > 0 else 0.0

    # 예측값 범위 제한: 또한 음수는 허용하지 않음
    y_pred_clipped = np.clip(np.asarray(y_pred, dtype=float), max(0.0, historical_min), historical_max)
    y_lower_clipped = np.clip(np.asarray(y_lower, dtype=float), max(0.0, historical_min), historical_max)
    y_upper_clipped = np.clip(np.asarray(y_upper, dtype=float), max(0.0, historical_min), historical_max)

    # 1. MAE (Mean Absolute Error) - 절대오차
    mae = float(np.mean(np.abs(np.asarray(y_true) - y_pred_clipped)))
    mae_score = max(0.0, 100.0 * (1.0 - mae / denom_mae))

    # 2. RMSE (Root Mean Squared Error) - 제곱근 오차
    rmse = float(np.sqrt(np.mean((np.asarray(y_true) - y_pred_clipped) ** 2)))
    rmse_score = max(0.0, 100.0 * (1.0 - rmse / denom_rmse))

    # 3. 신뢰구간 적절성 평가
    avg_width = float(np.mean(y_upper_clipped - y_lower_clipped))
    expected_width = max(1e-6, 2.0 * y_std)
    width_ratio = avg_width / expected_width if expected_width > 0 else 1.0
    # width_ratio <=0 는 비정상; guard
    if width_ratio <= 0:
        width_score = 0.0
    else:
        width_score = 100.0 * float(np.exp(-abs(np.log2(width_ratio))))

    # 3.2 실제값이 신뢰구간 안에 들어가는 비율
    in_interval = float(np.mean((y_true >= y_lower_clipped) & (y_true <= y_upper_clipped))) * 100.0
    # interval_score: 100 when in_interval == 95, decrease smoothly otherwise
    interval_score = 100.0 * float(np.exp(-abs(0.95 - (in_interval/100.0))))

    # 4. R-squared 계산 (보조 지표)
    ss_res = float(np.sum((y_true - y_pred_clipped) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # 5. 예측값의 변동성 체크
    pred_volatility = np.std(y_pred_clipped) / (np.mean(y_pred_clipped) + 1e-10)
    true_volatility = np.std(y_true) / (np.mean(y_true) + 1e-10)
    # guard ratio
    ratio = (pred_volatility / (true_volatility + 1e-10))
    volatility_score = 100.0 * float(np.exp(-abs(np.log2(ratio + 1e-10))))
    
    # 종합 신뢰도 점수 계산 (0~100)
    reliability_score = (
        mae_score * 0.25 +           # 절대오차 (25%)
        rmse_score * 0.25 +          # 제곱근 오차 (25%)
        width_score * 0.2 +          # 신뢰구간 너비 적절성 (20%)
        interval_score * 0.2 +       # 신뢰구간 포함률 (20%)
        volatility_score * 0.1       # 변동성 유사도 (10%)
    )
    
    return {
        "mae_score": mae_score,
        "rmse_score": rmse_score,
        "width_score": width_score,
        "interval_score": interval_score,
        "r2": r2,
        "volatility_score": volatility_score,
        "reliability_score": reliability_score
    }

def early_warning_rule(y: pd.Series) -> dict:
    # counts cannot be negative; clip series to >=0
    y = y.astype(float).clip(lower=0.0)

    if len(y) < (EWS_WINDOW_RECENT + EWS_WINDOW_BASE):
        return {"alert": False, "score": 0.0, "recent": np.nan, "baseline": np.nan, "sigma": np.nan}
    recent = float(np.mean(y[-EWS_WINDOW_RECENT:]))
    base_seg = y[-(EWS_WINDOW_RECENT + EWS_WINDOW_BASE):-EWS_WINDOW_RECENT]
    baseline = float(np.mean(base_seg))
    sigma = float(np.std(base_seg, ddof=1))
    alert = recent > (baseline + sigma)
    score = 0.0 if sigma == 0 else (recent - baseline) / sigma
    return {"alert": bool(alert), "score": score, "recent": recent, "baseline": baseline, "sigma": sigma}
