"""
운영 가드라인 (Guards)

데이터 품질 및 모델 안정성 체크:
1. 희소도 가드 - 0값이 과도한 시리즈 식별
2. 드리프트 가드 - 데이터 분포 변화 감지
3. 실측 누락 가드 - 불완전한 데이터 체크
4. 이상치 가드 - 극단값 감지

실행 예시:
    from src.guards import check_sparsity, check_drift, check_completeness
    
    is_sparse = check_sparsity(y)
    has_drift = check_drift(y)
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def check_sparsity(y: Union[np.ndarray, pd.Series],
                   threshold: float = 0.8,
                   return_stats: bool = False) -> Union[bool, Tuple[bool, Dict]]:
    """
    희소도 체크 - 0값 비율이 임계값을 초과하는지 확인
    
    Args:
        y: 시계열 데이터
        threshold: 0값 비율 임계값 (0~1)
        return_stats: 통계 정보 반환 여부
    
    Returns:
        희소 여부 (또는 희소 여부 + 통계 딕셔너리)
    """
    y = np.array(y)
    
    zero_count = (y == 0).sum()
    zero_ratio = zero_count / len(y)
    
    is_sparse = zero_ratio >= threshold
    
    if return_stats:
        stats_dict = {
            'zero_count': int(zero_count),
            'total_count': len(y),
            'zero_ratio': float(zero_ratio),
            'threshold': threshold,
            'is_sparse': is_sparse,
        }
        return is_sparse, stats_dict
    
    return is_sparse


def check_drift(y: Union[np.ndarray, pd.Series],
                window: int = 52,
                sigma_threshold: float = 3.0,
                return_stats: bool = False) -> Union[bool, Tuple[bool, Dict]]:
    """
    드리프트 가드 - 최근 데이터의 평균/분산이 과거와 유의미하게 다른지 확인
    
    Args:
        y: 시계열 데이터
        window: 최근 데이터 윈도우 크기
        sigma_threshold: 표준편차 배수 임계값
        return_stats: 통계 정보 반환 여부
    
    Returns:
        드리프트 존재 여부 (또는 존재 여부 + 통계 딕셔너리)
    """
    y = np.array(y)
    
    if len(y) <= window:
        # 데이터가 충분하지 않으면 드리프트 없음으로 간주
        if return_stats:
            return False, {'insufficient_data': True}
        return False
    
    # 최근 데이터와 과거 데이터 분리
    recent = y[-window:]
    historical = y[:-window]
    
    # 평균 변화 (표준편차 단위)
    hist_mean = historical.mean()
    hist_std = historical.std()
    
    if hist_std == 0:
        mean_shift = 0
    else:
        mean_shift = abs(recent.mean() - hist_mean) / hist_std
    
    # 분산 변화 (상대적 비율)
    hist_var = historical.var()
    recent_var = recent.var()
    
    if hist_var == 0:
        var_shift = 0
    else:
        var_shift = abs(recent_var - hist_var) / hist_var
    
    # 드리프트 판정
    has_drift = (mean_shift > sigma_threshold) or (var_shift > 1.0)
    
    if return_stats:
        stats_dict = {
            'mean_shift_sigma': float(mean_shift),
            'var_shift_ratio': float(var_shift),
            'recent_mean': float(recent.mean()),
            'historical_mean': float(hist_mean),
            'recent_var': float(recent_var),
            'historical_var': float(hist_var),
            'has_drift': has_drift,
        }
        return has_drift, stats_dict
    
    return has_drift


def check_completeness(y: Union[np.ndarray, pd.Series],
                      dates: Union[np.ndarray, pd.Series] = None,
                      expected_length: int = 52,
                      return_stats: bool = False) -> Union[bool, Tuple[bool, Dict]]:
    """
    데이터 완전성 체크 - 누락된 값이나 기간이 있는지 확인
    
    Args:
        y: 시계열 데이터
        dates: 날짜 정보 (선택)
        expected_length: 기대되는 데이터 길이 (주 단위)
        return_stats: 통계 정보 반환 여부
    
    Returns:
        완전 여부 (또는 완전 여부 + 통계 딕셔너리)
    """
    y = np.array(y)
    
    # NaN 체크
    nan_count = np.isnan(y).sum()
    
    # 길이 체크
    length_ok = len(y) >= expected_length
    
    # 날짜 연속성 체크 (날짜 정보가 있는 경우)
    has_gaps = False
    if dates is not None:
        dates = pd.to_datetime(pd.Series(dates))
        date_diffs = dates.diff().dt.days
        # 7일(1주) 이상 차이나는 경우 갭으로 간주
        has_gaps = (date_diffs > 10).any()
    
    is_complete = (nan_count == 0) and length_ok and (not has_gaps)
    
    if return_stats:
        stats_dict = {
            'nan_count': int(nan_count),
            'total_length': len(y),
            'expected_length': expected_length,
            'length_ok': length_ok,
            'has_gaps': has_gaps,
            'is_complete': is_complete,
        }
        return is_complete, stats_dict
    
    return is_complete


def check_outliers(y: Union[np.ndarray, pd.Series],
                   method: str = 'iqr',
                   threshold: float = 3.0,
                   return_indices: bool = False) -> Union[bool, Tuple[bool, np.ndarray]]:
    """
    이상치 가드 - 극단값 감지
    
    Args:
        y: 시계열 데이터
        method: 'iqr' (IQR 방법) 또는 'zscore' (Z-score)
        threshold: 임계값 (IQR: 배수, zscore: 표준편차)
        return_indices: 이상치 인덱스 반환 여부
    
    Returns:
        이상치 존재 여부 (또는 존재 여부 + 이상치 인덱스)
    """
    y = np.array(y)
    
    if method == 'iqr':
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outlier_indices = np.where((y < lower_bound) | (y > upper_bound))[0]
        
    elif method == 'zscore':
        mean = y.mean()
        std = y.std()
        
        if std == 0:
            outlier_indices = np.array([])
        else:
            z_scores = np.abs((y - mean) / std)
            outlier_indices = np.where(z_scores > threshold)[0]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    has_outliers = len(outlier_indices) > 0
    
    if return_indices:
        return has_outliers, outlier_indices
    
    return has_outliers


def check_seasonality_strength(y: Union[np.ndarray, pd.Series],
                               period: int = 52,
                               threshold: float = 0.3) -> Tuple[bool, float]:
    """
    계절성 강도 체크
    
    Args:
        y: 시계열 데이터
        period: 계절 주기
        threshold: 계절성 강도 임계값
    
    Returns:
        (강한 계절성 존재 여부, 계절성 강도)
    """
    y = np.array(y)
    
    if len(y) < period * 2:
        return False, 0.0
    
    # 계절성 분해 (간단한 방법)
    n_periods = len(y) // period
    
    seasonal_avg = np.zeros(period)
    seasonal_count = np.zeros(period)
    
    for i in range(len(y)):
        season_idx = i % period
        seasonal_avg[season_idx] += y[i]
        seasonal_count[season_idx] += 1
    
    seasonal_avg = np.where(seasonal_count > 0,
                           seasonal_avg / seasonal_count,
                           0)
    
    # 계절성 강도 = 계절 패턴의 변동 / 전체 변동
    seasonal_var = seasonal_avg.var()
    total_var = y.var()
    
    if total_var == 0:
        strength = 0.0
    else:
        strength = min(1.0, seasonal_var / total_var)
    
    has_strong_seasonality = strength >= threshold
    
    return has_strong_seasonality, strength


def run_all_guards(y: Union[np.ndarray, pd.Series],
                   dates: Union[np.ndarray, pd.Series] = None,
                   series_id: str = None) -> Dict:
    """
    모든 가드 체크 실행
    
    Args:
        y: 시계열 데이터
        dates: 날짜 정보 (선택)
        series_id: 시리즈 ID (로깅용)
    
    Returns:
        전체 가드 결과 딕셔너리
    """
    results = {
        'series_id': series_id,
    }
    
    # 1. 희소도
    is_sparse, sparse_stats = check_sparsity(y, return_stats=True)
    results['sparsity'] = sparse_stats
    
    # 2. 드리프트
    has_drift, drift_stats = check_drift(y, return_stats=True)
    results['drift'] = drift_stats
    
    # 3. 완전성
    is_complete, complete_stats = check_completeness(y, dates, return_stats=True)
    results['completeness'] = complete_stats
    
    # 4. 이상치
    has_outliers, outlier_indices = check_outliers(y, return_indices=True)
    results['outliers'] = {
        'has_outliers': has_outliers,
        'count': len(outlier_indices),
        'ratio': len(outlier_indices) / len(y) if len(y) > 0 else 0,
    }
    
    # 5. 계절성
    has_seasonality, strength = check_seasonality_strength(y)
    results['seasonality'] = {
        'has_strong_seasonality': has_seasonality,
        'strength': strength,
    }
    
    # 종합 판정
    results['recommendations'] = []
    
    if is_sparse:
        results['recommendations'].append('USE_NAIVE_MODEL')
    
    if has_drift:
        results['recommendations'].append('APPLY_SEASONAL_RECALIBRATION')
    
    if not is_complete:
        results['recommendations'].append('SKIP_RECONCILE')
    
    if has_outliers and outlier_indices.shape[0] / len(y) > 0.1:
        results['recommendations'].append('PREPROCESS_OUTLIERS')
    
    if not has_seasonality:
        results['recommendations'].append('USE_SIMPLE_MODEL')
    
    return results


def batch_guard_check(df: pd.DataFrame,
                     value_col: str = 'y',
                     date_col: str = 'week_end_date',
                     series_col: str = 'series_id') -> pd.DataFrame:
    """
    전체 시리즈에 대해 가드 체크 실행
    
    Args:
        df: 데이터프레임
        value_col: 값 컬럼명
        date_col: 날짜 컬럼명
        series_col: 시리즈 ID 컬럼명
    
    Returns:
        가드 체크 결과 데이터프레임
    """
    all_results = []
    
    for series_id in df[series_col].unique():
        series_data = df[df[series_col] == series_id].sort_values(date_col)
        
        y = series_data[value_col].values
        dates = series_data[date_col].values if date_col in series_data.columns else None
        
        results = run_all_guards(y, dates, series_id)
        
        # Flatten 결과
        flat_result = {
            'series_id': series_id,
            'is_sparse': results['sparsity']['is_sparse'],
            'zero_ratio': results['sparsity']['zero_ratio'],
            'has_drift': results['drift']['has_drift'],
            'mean_shift_sigma': results['drift']['mean_shift_sigma'],
            'is_complete': results['completeness']['is_complete'],
            'has_outliers': results['outliers']['has_outliers'],
            'outlier_ratio': results['outliers']['ratio'],
            'has_seasonality': results['seasonality']['has_strong_seasonality'],
            'seasonality_strength': results['seasonality']['strength'],
            'recommendations': ','.join(results['recommendations']),
        }
        
        all_results.append(flat_result)
    
    return pd.DataFrame(all_results)


def apply_guard_filters(df: pd.DataFrame,
                       guard_results: pd.DataFrame,
                       series_col: str = 'series_id') -> pd.DataFrame:
    """
    가드 결과를 기반으로 시리즈 필터링
    
    Args:
        df: 원본 데이터프레임
        guard_results: 가드 체크 결과
        series_col: 시리즈 ID 컬럼명
    
    Returns:
        필터링된 데이터프레임
    """
    # 문제가 있는 시리즈 제외
    problem_series = guard_results[
        (guard_results['is_sparse'] == True) |
        (guard_results['is_complete'] == False)
    ][series_col].values
    
    print(f"⚠️  {len(problem_series)}개 시리즈 제외 (희소 또는 불완전)")
    
    filtered_df = df[~df[series_col].isin(problem_series)].copy()
    
    return filtered_df
