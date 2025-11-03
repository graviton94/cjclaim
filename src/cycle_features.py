# src/cycle_features.py
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    from scipy.signal import find_peaks, peak_prominences, peak_widths
except Exception:
    # Provide clear runtime error if SciPy signal routines are required but missing
    def _missing_scipy_placeholder(*args, **kwargs):
        raise ImportError(
            "scipy.signal is required for peak detection. Please install scipy: `python -m pip install scipy`"
        )

    find_peaks = _missing_scipy_placeholder
    peak_prominences = _missing_scipy_placeholder
    peak_widths = _missing_scipy_placeholder

def compute_psi(ts: np.ndarray) -> np.ndarray:
    """
    Compute PSI (Periodic Seasonality Index) for a time series.
    Returns array of same length with PSI values (rolling calculation).
    """
    ts = np.asarray(ts, dtype=float)
    ts = np.clip(ts, 0.0, None)
    
    # 간단한 주기성 지수: 이동 평균 대비 변동성
    window = 12  # 약 3개월
    if len(ts) < window:
        return np.zeros(len(ts))
    
    psi_values = np.zeros(len(ts))
    for i in range(window, len(ts)):
        segment = ts[i-window:i]
        mean_val = np.mean(segment)
        std_val = np.std(segment)
        if mean_val > 0:
            psi_values[i] = std_val / mean_val
        else:
            psi_values[i] = 0.0
    
    return psi_values

def detect_peaks(ts: np.ndarray, min_prominence: float = 1.0) -> np.ndarray:
    """
    Detect peaks in time series and return binary flags.
    Returns array of 0s and 1s (1 = peak).
    """
    ts = np.asarray(ts, dtype=float)
    ts = np.clip(ts, 0.0, None)
    
    if len(ts) < 3:
        return np.zeros(len(ts), dtype=int)
    
    peaks_df = find_peaks_basic(ts, prominence=min_prominence)
    
    flags = np.zeros(len(ts), dtype=int)
    if len(peaks_df) > 0:
        flags[peaks_df['idx'].values] = 1
    
    return flags

def detect_changepoints(ts: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """
    Detect changepoints using simple threshold on differences.
    Returns array of 0s and 1s (1 = changepoint).
    """
    ts = np.asarray(ts, dtype=float)
    ts = np.clip(ts, 0.0, None)
    
    if len(ts) < 2:
        return np.zeros(len(ts), dtype=int)
    
    # 차분 계산
    diff = np.diff(ts, prepend=ts[0])
    
    # 이동 표준편차 계산
    window = 12
    flags = np.zeros(len(ts), dtype=int)
    
    for i in range(window, len(diff)):
        segment = diff[i-window:i]
        std_val = np.std(segment)
        if std_val > 0 and abs(diff[i]) > threshold * std_val:
            flags[i] = 1
    
    return flags

def compute_amplitude(ts: np.ndarray, window: int = 12) -> np.ndarray:
    """
    Compute rolling amplitude (standard deviation) of time series.
    Returns array of same length.
    """
    ts = np.asarray(ts, dtype=float)
    ts = np.clip(ts, 0.0, None)
    
    if len(ts) < window:
        return np.zeros(len(ts))
    
    amplitude_values = np.zeros(len(ts))
    for i in range(window, len(ts)):
        segment = ts[i-window:i]
        amplitude_values[i] = np.std(segment)
    
    return amplitude_values

def amplitude(ts: np.ndarray) -> float:
    """
    Compute the amplitude of a time series.
    Returns the difference between maximum and minimum values.
    """
    ts = np.asarray(ts, dtype=float)
    ts = np.clip(ts, 0.0, None)
    if len(ts) == 0:
        return 0.0
    return float(np.max(ts) - np.min(ts))

def _pad_to_len(arr: np.ndarray, n: int) -> np.ndarray:
    """길이를 n에 맞춤: 부족분은 NaN 패딩, 초과분은 절단, n==0이면 빈배열."""
    arr = np.asarray(arr)
    if n == 0:
        return np.asarray([], dtype=float)
    if arr.size == n:
        return arr
    if arr.size == 0:
        return np.full(n, np.nan, dtype=float)
    if arr.size > n:
        return arr[:n]
    return np.concatenate([arr, np.full(n - arr.size, np.nan, dtype=float)])

def find_peaks_basic(ts: np.ndarray,
                     height=None, distance=None,
                     prominence=None, width=None) -> pd.DataFrame:
    """
    ts에서 peak 탐지 후 idx/prominence/width를 동일 길이로 반환.
    prominence/width 누락 시 계산하거나 NaN으로 패딩.
    빈 결과(N==0)는 컬럼만 가진 빈 DataFrame 반환.
    """
    ts = np.asarray(ts, dtype=float)
    # counts cannot be negative
    ts = np.clip(ts, 0.0, None)
    peaks, props = find_peaks(ts, height=height, distance=distance,
                              prominence=prominence, width=width)
    n = len(peaks)

    # prominence 확보
    prominences = props.get("prominences", None)
    if prominences is None:
        prominences = peak_prominences(ts, peaks)[0] if n > 0 else np.asarray([])

    # width 확보 (rel_height=0.5 기본)
    widths = props.get("widths", None)
    if widths is None:
        widths = peak_widths(ts, peaks, rel_height=0.5)[0] if n > 0 else np.asarray([])

    prominences = _pad_to_len(prominences, n)
    widths      = _pad_to_len(widths, n)

    if n == 0:
        return pd.DataFrame(columns=["idx", "prominence", "width"])

    return pd.DataFrame({
        "idx": peaks.astype(int),
        "prominence": prominences,
        "width": widths
    })
