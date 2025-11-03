"""
메트릭 계산 유틸리티
MAPE, MASE, Bias 등 시계열 예측 성능 지표 계산
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple


def compute_mape(y_true: Union[np.ndarray, pd.Series], 
                 y_pred: Union[np.ndarray, pd.Series],
                 epsilon: float = 1e-10) -> float:
    """
    Mean Absolute Percentage Error 계산
    
    Args:
        y_true: 실제값
        y_pred: 예측값
        epsilon: 0으로 나누기 방지용 작은 값
    
    Returns:
        MAPE 값 (0~1 스케일)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 0이 아닌 값들만 사용
    mask = y_true != 0
    if not mask.any():
        return np.nan
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon)))
    return mape


def compute_mase(y_true: Union[np.ndarray, pd.Series],
                 y_pred: Union[np.ndarray, pd.Series],
                 y_train: Union[np.ndarray, pd.Series],
                 seasonal_period: int = 52) -> float:
    """
    Mean Absolute Scaled Error 계산
    
    Args:
        y_true: 실제값
        y_pred: 예측값
        y_train: 학습 데이터 (MAE 스케일 계산용)
        seasonal_period: 계절 주기 (기본: 52주)
    
    Returns:
        MASE 값 (1.0 = Seasonal Naive와 동일 성능)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_train = np.array(y_train)
    
    # MAE of forecast
    mae_forecast = np.mean(np.abs(y_true - y_pred))
    
    # MAE of seasonal naive forecast on training data
    if len(y_train) <= seasonal_period:
        # 충분한 데이터가 없으면 일반 naive 사용
        mae_naive = np.mean(np.abs(np.diff(y_train)))
    else:
        naive_errors = y_train[seasonal_period:] - y_train[:-seasonal_period]
        mae_naive = np.mean(np.abs(naive_errors))
    
    if mae_naive == 0:
        return np.nan
    
    mase = mae_forecast / mae_naive
    return mase


def compute_bias(y_true: Union[np.ndarray, pd.Series],
                 y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Bias (평균 오차율) 계산
    양수: 과대예측, 음수: 과소예측
    
    Args:
        y_true: 실제값
        y_pred: 예측값
    
    Returns:
        Bias 값 (-1~1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    total_true = np.sum(y_true)
    if total_true == 0:
        return np.nan
    
    bias = (np.sum(y_pred) - total_true) / total_true
    return bias


def compute_mae(y_true: Union[np.ndarray, pd.Series],
                y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Mean Absolute Error 계산"""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def compute_rmse(y_true: Union[np.ndarray, pd.Series],
                 y_pred: Union[np.ndarray, pd.Series]) -> float:
    """Root Mean Squared Error 계산"""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def compute_all_metrics(y_true: Union[np.ndarray, pd.Series],
                        y_pred: Union[np.ndarray, pd.Series],
                        y_train: Union[np.ndarray, pd.Series] = None,
                        seasonal_period: int = 52) -> dict:
    """
    모든 주요 메트릭을 한번에 계산
    
    Args:
        y_true: 실제값
        y_pred: 예측값
        y_train: 학습 데이터 (MASE 계산용, 선택)
        seasonal_period: 계절 주기
    
    Returns:
        메트릭 딕셔너리
    """
    metrics = {
        'mape': compute_mape(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'rmse': compute_rmse(y_true, y_pred),
        'bias': compute_bias(y_true, y_pred),
    }
    
    if y_train is not None:
        metrics['mase'] = compute_mase(y_true, y_pred, y_train, seasonal_period)
    
    return metrics


def compute_metrics_by_group(df: pd.DataFrame,
                             y_true_col: str = 'y_true',
                             y_pred_col: str = 'y_pred',
                             group_cols: list = None,
                             y_train: pd.DataFrame = None,
                             train_group_col: str = None) -> pd.DataFrame:
    """
    그룹별로 메트릭 계산
    
    Args:
        df: 예측 결과 데이터프레임
        y_true_col: 실제값 컬럼명
        y_pred_col: 예측값 컬럼명
        group_cols: 그룹화 컬럼 리스트
        y_train: 학습 데이터 (MASE 계산용)
        train_group_col: 학습 데이터 그룹 컬럼
    
    Returns:
        그룹별 메트릭 데이터프레임
    """
    if group_cols is None:
        group_cols = ['series_id']
    
    results = []
    
    for group_keys, group_df in df.groupby(group_cols):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        
        y_true = group_df[y_true_col].values
        y_pred = group_df[y_pred_col].values
        
        # 학습 데이터 찾기 (MASE용)
        train_data = None
        if y_train is not None and train_group_col is not None:
            if len(group_cols) == 1:
                train_data = y_train[y_train[train_group_col] == group_keys[0]]
            else:
                # 복수 그룹 컬럼 처리
                mask = True
                for col, val in zip(group_cols, group_keys):
                    mask &= (y_train[col] == val)
                train_data = y_train[mask]
            
            if len(train_data) > 0:
                train_values = train_data[y_true_col].values
            else:
                train_values = None
        else:
            train_values = None
        
        # 메트릭 계산
        metrics = compute_all_metrics(y_true, y_pred, train_values)
        
        # 결과 저장
        result = dict(zip(group_cols, group_keys))
        result.update(metrics)
        result['n_observations'] = len(y_true)
        
        results.append(result)
    
    return pd.DataFrame(results)


def identify_poor_performers(metrics_df: pd.DataFrame,
                            mape_threshold: float = 0.20,
                            bias_threshold: float = 0.05,
                            mase_threshold: float = 1.5) -> pd.DataFrame:
    """
    성능이 낮은 시리즈 식별
    
    Args:
        metrics_df: 메트릭 데이터프레임
        mape_threshold: MAPE 임계값
        bias_threshold: Bias 절대값 임계값
        mase_threshold: MASE 임계값
    
    Returns:
        튜닝 후보 시리즈 데이터프레임
    """
    candidates = metrics_df[
        (metrics_df['mape'] > mape_threshold) |
        (metrics_df['bias'].abs() > bias_threshold) |
        (metrics_df['mase'] > mase_threshold)
    ].copy()
    
    # 우선순위 스코어 계산 (높을수록 문제가 심각)
    candidates['priority_score'] = (
        (candidates['mape'] / mape_threshold) * 0.4 +
        (candidates['bias'].abs() / bias_threshold) * 0.3 +
        (candidates['mase'] / mase_threshold) * 0.3
    )
    
    candidates = candidates.sort_values('priority_score', ascending=False)
    
    return candidates
