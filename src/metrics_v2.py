"""
Quality Metrics v2 - 3-Metric KPI System
=========================================
1. WMAPE (Weighted MAPE): Σ|e| / Σy[y>0]
2. SMAPE (Symmetric MAPE): mean(|e| / ((|y|+|ŷ|)/2))
3. Bias: mean(e) / mean(y[y>0])

Performance Levels:
- WMAPE: Excellent <20%, Good 20-50%, Fair 50-100%, Poor >100%
- SMAPE: Excellent <15%, Good 15-30%, Fair 30-50%, Poor >50%
- Bias: Excellent ±5%, Good ±10%, Fair ±20%, Poor >±20%
=========================================
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def calculate_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted MAPE = Σ|e| / Σy[y>0]
    
    Only considers periods where actual > 0 to avoid division issues
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        WMAPE percentage (0-100+)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Filter positive actuals
    positive_mask = y_true > 0
    
    if not positive_mask.any():
        return np.nan
    
    y_true_pos = y_true[positive_mask]
    y_pred_pos = y_pred[positive_mask]
    
    errors = np.abs(y_true_pos - y_pred_pos)
    wmape = (errors.sum() / y_true_pos.sum()) * 100
    
    return float(wmape)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric MAPE = mean(|e| / ((|y|+|ŷ|)/2))
    
    Handles zero values better than standard MAPE
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        SMAPE percentage (0-200, typically <100)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    errors = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Avoid division by zero
    valid_mask = denominator > 0
    
    if not valid_mask.any():
        return np.nan
    
    smape = (errors[valid_mask] / denominator[valid_mask]).mean() * 100
    
    return float(smape)


def calculate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Bias = mean(e) / mean(y[y>0])
    
    Positive: Over-prediction
    Negative: Under-prediction
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Bias percentage (-100 to +100+)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Filter positive actuals
    positive_mask = y_true > 0
    
    if not positive_mask.any():
        return np.nan
    
    y_true_pos = y_true[positive_mask]
    y_pred_pos = y_pred[positive_mask]
    
    errors = y_pred_pos - y_true_pos  # Signed errors
    bias = (errors.mean() / y_true_pos.mean()) * 100
    
    return float(bias)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all 3 metrics at once
    
    Returns:
        {
            'wmape': float,
            'smape': float,
            'bias': float,
            'mae': float,
            'rmse': float
        }
    """
    wmape = calculate_wmape(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)
    bias = calculate_bias(y_true, y_pred)
    
    # Bonus: MAE and RMSE
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    return {
        'wmape': wmape,
        'smape': smape,
        'bias': bias,
        'mae': mae,
        'rmse': rmse
    }


def get_performance_level(metric_name: str, value: float) -> str:
    """
    Get performance level label
    
    Args:
        metric_name: 'wmape', 'smape', or 'bias'
        value: Metric value
    
    Returns:
        'EXCELLENT', 'GOOD', 'FAIR', or 'POOR'
    """
    if pd.isna(value):
        return 'UNKNOWN'
    
    if metric_name == 'wmape':
        if value < 20:
            return 'EXCELLENT'
        elif value < 50:
            return 'GOOD'
        elif value < 100:
            return 'FAIR'
        else:
            return 'POOR'
    
    elif metric_name == 'smape':
        if value < 15:
            return 'EXCELLENT'
        elif value < 30:
            return 'GOOD'
        elif value < 50:
            return 'FAIR'
        else:
            return 'POOR'
    
    elif metric_name == 'bias':
        abs_bias = abs(value)
        if abs_bias < 5:
            return 'EXCELLENT'
        elif abs_bias < 10:
            return 'GOOD'
        elif abs_bias < 20:
            return 'FAIR'
        else:
            return 'POOR'
    
    return 'UNKNOWN'


def format_metrics_report(metrics: Dict[str, float]) -> str:
    """
    Format metrics as a human-readable report
    
    Args:
        metrics: Output from calculate_all_metrics
    
    Returns:
        Formatted string report
    """
    wmape_level = get_performance_level('wmape', metrics['wmape'])
    smape_level = get_performance_level('smape', metrics['smape'])
    bias_level = get_performance_level('bias', metrics['bias'])
    
    report = f"""
Quality Metrics Report
{'='*50}
WMAPE: {metrics['wmape']:.2f}% ({wmape_level})
SMAPE: {metrics['smape']:.2f}% ({smape_level})
Bias:  {metrics['bias']:+.2f}% ({bias_level})
{'='*50}
MAE:   {metrics['mae']:.2f}
RMSE:  {metrics['rmse']:.2f}
"""
    return report


def cross_validate_metrics(y: np.ndarray, 
                            model_func,
                            n_splits: int = 3,
                            test_size: int = 6) -> pd.DataFrame:
    """
    Perform rolling cross-validation and compute metrics for each fold
    
    Args:
        y: Time series data
        model_func: Function that takes (train_data) and returns fitted model
        n_splits: Number of CV splits
        test_size: Test period size (months)
    
    Returns:
        DataFrame with metrics for each fold
    """
    results = []
    
    for fold in range(n_splits):
        # Split point
        split_idx = len(y) - (n_splits - fold) * test_size
        
        if split_idx < 12:  # Need at least 12 months for training
            continue
        
        y_train = y[:split_idx]
        y_test = y[split_idx:split_idx + test_size]
        
        if len(y_test) == 0:
            continue
        
        try:
            # Fit model
            model = model_func(y_train)
            
            # Forecast
            forecast = model.forecast(steps=len(y_test))
            
            # Calculate metrics
            metrics = calculate_all_metrics(y_test, forecast)
            metrics['fold'] = fold + 1
            metrics['train_size'] = len(y_train)
            metrics['test_size'] = len(y_test)
            
            results.append(metrics)
        except Exception as e:
            print(f"[WARNING] Fold {fold+1} failed: {e}")
            continue
    
    return pd.DataFrame(results)


def aggregate_cv_metrics(cv_df: pd.DataFrame) -> Dict[str, float]:
    """
    Aggregate cross-validation metrics across folds
    
    Args:
        cv_df: Output from cross_validate_metrics
    
    Returns:
        Mean metrics across folds
    """
    if len(cv_df) == 0:
        return {
            'wmape': np.nan,
            'smape': np.nan,
            'bias': np.nan,
            'mae': np.nan,
            'rmse': np.nan
        }
    
    return {
        'wmape': cv_df['wmape'].mean(),
        'smape': cv_df['smape'].mean(),
        'bias': cv_df['bias'].mean(),
        'mae': cv_df['mae'].mean(),
        'rmse': cv_df['rmse'].mean()
    }


if __name__ == "__main__":
    # Test example
    np.random.seed(42)
    
    y_true = np.random.poisson(10, 50) + 5
    y_pred = y_true + np.random.normal(0, 3, 50)
    
    metrics = calculate_all_metrics(y_true, y_pred)
    print(format_metrics_report(metrics))
    
    print("\nPerformance Levels:")
    for metric_name in ['wmape', 'smape', 'bias']:
        level = get_performance_level(metric_name, metrics[metric_name])
        print(f"{metric_name.upper()}: {level}")
