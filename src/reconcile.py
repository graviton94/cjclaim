__all__ = ["reconcile_month"]
"""
Reconcile 모듈: 예측 보정 로직

경량 보정 기법:
1. Bias Map 보정 - 주차별/월별 편향 패턴 보정
2. Seasonal Recalibration - 계절성 재추정
3. Changepoint-aware Hold - 변화점 감지 시 폴백 유지

실행 예시:
    from src.reconcile import BiasCorrector, SeasonalRecalibrator
    
    corrector = BiasCorrector()
    y_adjusted = corrector.fit_transform(y_pred, y_true, week_info)
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Tuple
import warnings
from scipy import stats

warnings.filterwarnings('ignore')


class BiasCorrector:
    """
    주차별/월별 편향 보정
    
    과거 예측 오차의 주기적 패턴을 학습하여
    미래 예측을 보정
    """
    
    def __init__(self, method: str = 'weekly', window: int = 52):
        """
        Args:
            method: 'weekly' 또는 'monthly'
            window: 학습에 사용할 최근 주차 수
        """
        self.method = method
        self.window = window
        self.bias_map_ = None
    
    def fit(self, 
            y_true: Union[np.ndarray, pd.Series],
            y_pred: Union[np.ndarray, pd.Series],
            week_info: Union[np.ndarray, pd.Series] = None) -> 'BiasCorrector':
        """
        편향 맵 학습
        
        Args:
            y_true: 실제값
            y_pred: 예측값
            week_info: 주차 정보 (1-52) 또는 월 정보 (1-12)
        
        Returns:
            self
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 편향 계산 (실제 - 예측)
        bias = y_true - y_pred
        
        if week_info is None:
            # 주차 정보가 없으면 순차적으로 할당
            week_info = np.arange(len(bias)) % 52 + 1
        else:
            week_info = np.array(week_info)
        
        # 주차별 평균 편향 계산
        bias_df = pd.DataFrame({
            'week': week_info,
            'bias': bias
        })
        
        if self.method == 'weekly':
            self.bias_map_ = bias_df.groupby('week')['bias'].mean().to_dict()
        elif self.method == 'monthly':
            # 주차를 월로 변환 (대략 4.3주 = 1개월)
            bias_df['month'] = ((bias_df['week'] - 1) // 4.3).astype(int) + 1
            self.bias_map_ = bias_df.groupby('month')['bias'].mean().to_dict()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def transform(self, 
                  y_pred: Union[np.ndarray, pd.Series],
                  week_info: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        학습된 편향 맵으로 예측값 보정
        
        Args:
            y_pred: 예측값
            week_info: 주차/월 정보
        
        Returns:
            보정된 예측값
        """
        if self.bias_map_ is None:
            raise ValueError("BiasCorrector must be fitted before transform")
        
        y_pred = np.array(y_pred)
        week_info = np.array(week_info)
        
        if self.method == 'monthly':
            week_info = ((week_info - 1) // 4.3).astype(int) + 1
        
        # 각 주차/월에 해당하는 편향 적용
        y_adjusted = y_pred.copy()
        for i, week in enumerate(week_info):
            bias_adj = self.bias_map_.get(week, 0.0)
            y_adjusted[i] += bias_adj
        
        return y_adjusted
    
    def fit_transform(self,
                      y_pred: Union[np.ndarray, pd.Series],
                      y_true: Union[np.ndarray, pd.Series],
                      week_info: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(y_true, y_pred, week_info)
        return self.transform(y_pred, week_info)


class SeasonalRecalibrator:
    """
    계절성 재보정
    
    최근 데이터를 사용하여 계절성 성분만 재추정
    """
    
    def __init__(self, seasonal_period: int = 52, recent_years: int = 2):
        """
        Args:
            seasonal_period: 계절 주기 (기본: 52주)
            recent_years: 사용할 최근 연도 수
        """
        self.seasonal_period = seasonal_period
        self.recent_years = recent_years
        self.seasonal_factors_ = None
    
    def fit(self, y: Union[np.ndarray, pd.Series]) -> 'SeasonalRecalibrator':
        """
        계절성 패턴 학습
        
        Args:
            y: 시계열 데이터 (최근 데이터 사용)
        
        Returns:
            self
        """
        y = np.array(y)
        
        # 최근 데이터만 사용
        recent_window = self.seasonal_period * self.recent_years
        if len(y) > recent_window:
            y = y[-recent_window:]
        
        # 계절성 분해 (간단한 평균 방법)
        n_seasons = len(y) // self.seasonal_period
        
        if n_seasons < 1:
            # 데이터가 충분하지 않으면 계절성 없음
            self.seasonal_factors_ = np.ones(self.seasonal_period)
            return self
        
        # 각 계절 위치의 평균값 계산
        seasonal_avg = np.zeros(self.seasonal_period)
        seasonal_count = np.zeros(self.seasonal_period)
        
        for i in range(len(y)):
            season_idx = i % self.seasonal_period
            seasonal_avg[season_idx] += y[i]
            seasonal_count[season_idx] += 1
        
        # 평균 계산
        seasonal_avg = np.where(seasonal_count > 0, 
                               seasonal_avg / seasonal_count,
                               0)
        
        # 전체 평균으로 정규화
        overall_mean = seasonal_avg.mean()
        if overall_mean > 0:
            self.seasonal_factors_ = seasonal_avg / overall_mean
        else:
            self.seasonal_factors_ = np.ones(self.seasonal_period)
        
        return self
    
    def transform(self, y_pred: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        학습된 계절성 패턴을 예측값에 적용
        
        Args:
            y_pred: 예측값
        
        Returns:
            재보정된 예측값
        """
        if self.seasonal_factors_ is None:
            raise ValueError("SeasonalRecalibrator must be fitted before transform")
        
        y_pred = np.array(y_pred)
        
        # 각 예측값에 계절 인자 적용
        y_recalibrated = np.zeros_like(y_pred)
        for i in range(len(y_pred)):
            season_idx = i % self.seasonal_period
            y_recalibrated[i] = y_pred[i] * self.seasonal_factors_[season_idx]
        
        return y_recalibrated
    
    def fit_transform(self, 
                      y_train: Union[np.ndarray, pd.Series],
                      y_pred: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Fit on training data and transform predictions"""
        self.fit(y_train)
        return self.transform(y_pred)


class ChangepointDetector:
    """
    변화점 감지
    
    구조적 변화가 발생한 구간을 탐지하여
    해당 구간에서는 폴백 모델 사용
    """
    
    def __init__(self, method: str = 'cusum', threshold: float = 3.0):
        """
        Args:
            method: 'cusum', 'ruptures', 또는 'statistical'
            threshold: 감지 임계값 (표준편차 단위)
        """
        self.method = method
        self.threshold = threshold
        self.changepoints_ = None
    
    def detect(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        변화점 감지
        
        Args:
            y: 시계열 데이터
        
        Returns:
            변화점 인덱스 배열
        """
        y = np.array(y)
        
        if self.method == 'cusum':
            self.changepoints_ = self._detect_cusum(y)
        elif self.method == 'ruptures':
            self.changepoints_ = self._detect_ruptures(y)
        elif self.method == 'statistical':
            self.changepoints_ = self._detect_statistical(y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self.changepoints_
    
    def _detect_cusum(self, y: np.ndarray) -> np.ndarray:
        """CUSUM 방법으로 변화점 감지"""
        # 평균과 표준편차 계산
        mean = np.mean(y)
        std = np.std(y)
        
        if std == 0:
            return np.array([])
        
        # CUSUM 통계량
        cusum_pos = np.zeros(len(y))
        cusum_neg = np.zeros(len(y))
        
        for i in range(1, len(y)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + (y[i] - mean) / std - 0.5)
            cusum_neg[i] = max(0, cusum_neg[i-1] - (y[i] - mean) / std - 0.5)
        
        # 임계값 초과 지점
        changepoints = np.where((cusum_pos > self.threshold) | 
                               (cusum_neg > self.threshold))[0]
        
        return changepoints
    
    def _detect_ruptures(self, y: np.ndarray) -> np.ndarray:
        """Ruptures 라이브러리 사용 (선택적)"""
        try:
            import ruptures as rpt
            algo = rpt.Pelt(model="rbf").fit(y)
            changepoints = algo.predict(pen=10)
            return np.array(changepoints[:-1])  # 마지막 인덱스 제외
        except ImportError:
            # ruptures가 없으면 statistical 방법 사용
            return self._detect_statistical(y)
    
    def _detect_statistical(self, y: np.ndarray, window: int = 10) -> np.ndarray:
        """통계적 방법으로 변화점 감지 (이동 평균/분산 변화)"""
        if len(y) < window * 2:
            return np.array([])
        
        changepoints = []
        
        for i in range(window, len(y) - window):
            before = y[i-window:i]
            after = y[i:i+window]
            
            # t-검정으로 평균 차이 검정
            t_stat, p_value = stats.ttest_ind(before, after)
            
            if p_value < 0.01:  # 유의수준 1%
                changepoints.append(i)
        
        return np.array(changepoints)
    
    def get_stable_regions(self, length: int, margin: int = 5) -> list:
        """
        안정적인 구간 반환 (변화점이 없는 구간)
        
        Args:
            length: 전체 시계열 길이
            margin: 변화점 주변 제외 마진
        
        Returns:
            [(start, end), ...] 형태의 안정 구간 리스트
        """
        if self.changepoints_ is None or len(self.changepoints_) == 0:
            return [(0, length)]
        
        stable_regions = []
        
        # 첫 변화점 이전
        if self.changepoints_[0] > margin:
            stable_regions.append((0, self.changepoints_[0] - margin))
        
        # 변화점 사이
        for i in range(len(self.changepoints_) - 1):
            start = self.changepoints_[i] + margin
            end = self.changepoints_[i+1] - margin
            if end > start:
                stable_regions.append((start, end))
        
        # 마지막 변화점 이후
        if self.changepoints_[-1] + margin < length:
            stable_regions.append((self.changepoints_[-1] + margin, length))
        
        return stable_regions


def apply_reconciliation(y_pred: Union[np.ndarray, pd.Series],
                        y_train: Union[np.ndarray, pd.Series],
                        y_true: Union[np.ndarray, pd.Series] = None,
                        week_info: Union[np.ndarray, pd.Series] = None,
                        apply_bias: bool = True,
                        apply_seasonal: bool = True,
                        detect_changepoints: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    통합 보정 파이프라인
    
    Args:
        y_pred: 예측값
        y_train: 학습 데이터
        y_true: 실측값 (bias 보정용, 선택)
        week_info: 주차 정보
        apply_bias: Bias 보정 적용 여부
        apply_seasonal: 계절성 재보정 적용 여부
        detect_changepoints: 변화점 감지 여부
    
    Returns:
        (보정된 예측값, 메타정보 딕셔너리)
    """
    y_adjusted = np.array(y_pred).copy()
    metadata = {
        'bias_adj': False,
        'seasonal_recal': False,
        'changepoints': [],
    }
    
    # 1. 변화점 감지
    changepoints = []
    if detect_changepoints:
        detector = ChangepointDetector(method='statistical')
        changepoints = detector.detect(y_train)
        metadata['changepoints'] = changepoints.tolist()
        
        if len(changepoints) > 0:
            print(f"  ⚠️  {len(changepoints)}개 변화점 감지")
    
    # 2. Bias 보정
    if apply_bias and y_true is not None:
        try:
            corrector = BiasCorrector(method='weekly')
            
            # 변화점이 없는 안정 구간에서만 학습
            if len(changepoints) > 0:
                detector = ChangepointDetector()
                detector.changepoints_ = changepoints
                stable_regions = detector.get_stable_regions(len(y_train))
                
                # 안정 구간 데이터 추출
                stable_indices = []
                for start, end in stable_regions:
                    stable_indices.extend(range(start, end))
                
                if len(stable_indices) > 10:
                    corrector.fit(
                        y_true[stable_indices],
                        y_pred[stable_indices],
                        week_info[stable_indices] if week_info is not None else None
                    )
                    y_adjusted = corrector.transform(y_adjusted, week_info)
                    metadata['bias_adj'] = True
            else:
                y_adjusted = corrector.fit_transform(y_adjusted, y_true, week_info)
                metadata['bias_adj'] = True
        except Exception as e:
            print(f"  ⚠️  Bias 보정 실패: {e}")
    
    # 3. 계절성 재보정
    if apply_seasonal:
        try:
            recalibrator = SeasonalRecalibrator(recent_years=2)
            y_adjusted = recalibrator.fit_transform(y_train, y_adjusted)
            metadata['seasonal_recal'] = True
        except Exception as e:
            print(f"  ⚠️  계절성 재보정 실패: {e}")
    
    return y_adjusted, metadata


def reconcile_month(forecast_parquet: str, train_parquet: str, out_parquet: str) -> dict:
    """
    S6: 월별 예측 결과 보정 및 저장
    Args:
        forecast_parquet: 예측 결과 parquet 경로
        train_parquet: 학습/실측 데이터 parquet 경로
        out_parquet: 보정 결과 저장 경로
    Returns:
        {"reconciled": out_parquet, "series_count": int}
    """
    import pandas as pd
    from pathlib import Path
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    # 1) 예측 결과 로드
    fc = pd.read_parquet(forecast_parquet)
    # 2) 실측 데이터 로드 (파일 없으면 빈 DataFrame)
    import os
    if not os.path.exists(train_parquet):
        tr = pd.DataFrame(columns=["series_id","y","count","발생건수"])
    else:
        tr = pd.read_parquet(train_parquet)
    # 3) series_id 기준 그룹화 및 보정 (월별 기준)
    rows = []
    for sid, g in fc.groupby("series_id", sort=False):
        # 실측 데이터 매칭
        g_tr = tr[tr["series_id"] == sid]
        y_pred = g["predicted_value"].values
        # value 컬럼 우선순위: y > count > 발생건수
        if "y" in g_tr.columns:
            y_train = g_tr["y"].values
        elif "count" in g_tr.columns:
            y_train = g_tr["count"].values
        elif "발생건수" in g_tr.columns:
            y_train = g_tr["발생건수"].values
        else:
            y_train = None
        month_info = g["forecast_month"].values if "forecast_month" in g.columns else None
        # 보정 적용
        if y_train is not None and len(y_train) > 0:
            y_adj, meta = apply_reconciliation(y_pred, y_train, None, month_info)
        else:
            y_adj, meta = y_pred, {}
        for i, row in g.iterrows():
            rows.append({
                "series_id": sid,
                "forecast_month": row["forecast_month"] if "forecast_month" in row else None,
                "predicted_value": float(max(0.0, y_adj[i - g.index[0]] if i - g.index[0] < len(y_adj) else 0.0)),
                "lower_bound": float(max(0.0, row["lower_bound"])),
                "upper_bound": float(max(0.0, row["upper_bound"])),
            })
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_parquet, index=False)
    if "series_id" in out_df.columns:
        series_count = int(out_df["series_id"].nunique())
    else:
        series_count = 0
    return {"reconciled": out_parquet, "series_count": series_count}
