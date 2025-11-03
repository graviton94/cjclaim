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

def compute_psi(ts: np.ndarray) -> float:
    """
    Compute PSI (Peak Significance Index) for a time series.
    PSI measures the significance of peaks in a time series.
    """
    peaks = find_peaks_basic(ts)
    if peaks.empty:
        return np.nan
    
    mean_prominence = peaks["prominence"].mean()
    mean_width = peaks["width"].mean()
    
    if mean_width == 0:
        return np.nan
        
    return mean_prominence / mean_width

def amplitude(ts: np.ndarray) -> float:
    """
    Compute the amplitude of a time series.
    Returns the difference between maximum and minimum values.
    """
    if len(ts) == 0:
        return 0.0
    return np.max(ts) - np.min(ts)

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
