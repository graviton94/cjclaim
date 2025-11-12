"""
EWS (Early Warning System) v2 - 5-Factor Scoring with Weight Learning
================================================================================
5-Factor Model:
  F1: Growth Ratio      - mean(ŷ[t+1:t+h]) / mean(y[t-12:t-1])
  F2: Confidence        - 0.5·π_compression + 0.5·coverage_80
  F3: Seasonality       - 1 - Var(resid) / Var(y)
  F4: Amplitude         - (max_season - min_season) / mean(y)
  F5: Rising-Inflection - 0.5·norm(accel) + 0.5·cp_prob

Combined Score: EWS = Σ(w_i · norm(F_i))
Weight Learning: Logistic Regression with L2 regularization
Validation: Rolling 3-fold, F1 & PR-AUC maximization
================================================================================
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import warnings
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, auc
import ruptures as rpt
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings('ignore')


class EWSScorer:
    def compute_amplitude(self, historical_data: np.ndarray, period: int = 12) -> float:
        """
        진폭(f4_ampl) 계산: 분산 가드, 표준화, 클램프 적용
        - 단일값/저분산 시 0.0, 정상 시 0~1 사이로 안정화
        """
        # Be more permissive: allow amplitude estimation with at least one seasonal cycle
        hist = np.asarray(historical_data, dtype=float)
        if hist.size < period:
            return 0.0

        std = float(np.std(hist))
        if std < 1e-8:
            return 0.0

        mean_y = float(np.mean(hist))
        if mean_y < 1e-6:
            return 0.0

        # Try STL first with seasonal=period (more standard) and robust fitting
        try:
            stl = STL(hist, period=period, robust=True)
            result = stl.fit()
            seasonal = np.asarray(result.seasonal)

            # Build seasonal pattern by averaging same-month positions across cycles
            n_cycles = max(1, len(seasonal) // period)
            seasonal_pattern = np.zeros(period)
            counts = np.zeros(period)
            for idx, val in enumerate(seasonal):
                seasonal_pattern[idx % period] += val
                counts[idx % period] += 1
            # avoid division by zero
            seasonal_pattern = seasonal_pattern / np.maximum(counts, 1)

            amp_raw = (seasonal_pattern.max() - seasonal_pattern.min())
            amplitude = float(np.clip(amp_raw / max(mean_y, 1e-9), 0.0, 1.0))

            # If STL returned a flat seasonal (no variation), fall back to simple monthly averages
            if np.isclose(amp_raw, 0.0):
                raise ValueError('flat_seasonal_from_stl')

            return amplitude

        except Exception:
            # Fallback: compute monthly (periodic) averages directly from raw history
            try:
                seasonal_pattern = np.zeros(period)
                counts = np.zeros(period)
                for idx, val in enumerate(hist):
                    seasonal_pattern[idx % period] += val
                    counts[idx % period] += 1
                seasonal_pattern = seasonal_pattern / np.maximum(counts, 1)
                amp_raw = seasonal_pattern.max() - seasonal_pattern.min()
                amplitude = float(np.clip(amp_raw / max(mean_y, 1e-9), 0.0, 1.0))
                return amplitude
            except Exception:
                return 0.0

    @staticmethod
    def to_yyyymm(year: int, month: int) -> str:
        """yyyymm 문자열 생성"""
        return f"{year:04d}{month:02d}"

    @staticmethod
    def month_seq(start_year: int, start_month: int, n: int = 6) -> list:
        """start_year, start_month부터 n개월 시퀀스 반환"""
        seq = []
        y, m = start_year, start_month
        for _ in range(n):
            seq.append(EWSScorer.to_yyyymm(y, m))
            m += 1
            if m > 12:
                m = 1
                y += 1
        return seq

    @staticmethod
    def pick_pred_col(df: pd.DataFrame) -> str:
        """예측값 컬럼 자동 선택"""
        for c in ["yhat", "predicted_value", "forecast", "y_pred"]:
            if c in df.columns:
                return c
        raise Exception("No prediction column found")

    def extract_forecast_window(self, df: pd.DataFrame, year: int, month: int, window: int = 6) -> Tuple[list, list]:
        """
        예측 윈도우 추출: yyyymm 생성, 오름차순 정렬, 중복 제거, 결측값 삭제, 부족하면 빈 리스트 처리
        """
        pred_col = self.pick_pred_col(df)
        months = []
        values = []
        # yyyymm 생성
        if "year" in df.columns and "month" in df.columns:
            df["yyyymm"] = [self.to_yyyymm(y, m) for y, m in zip(df["year"], df["month"])]
        elif "yyyymm" not in df.columns:
            raise Exception("No year/month/yyyymm column")
        # 윈도우 시퀀스
        seq = self.month_seq(year, month, window)
        # 추출 및 정제
        for ym in seq:
            rows = df[df["yyyymm"] == ym]
            if not rows.empty:
                v = rows[pred_col].dropna().unique()
                if len(v) > 0:
                    months.append(ym)
                    values.append(float(v[0]))
        # 오름차순, 중복 제거
        months, values = zip(*sorted(set(zip(months, values)))) if months else ([],[])
        # 부족하면 빈 리스트 처리
        if len(months) < window:
            return [], []
        return list(months), list(values)
    """5-Factor EWS Scoring Engine with Weight Learning"""
    
    def __init__(self, 
                 weights: Optional[Dict[str, float]] = None,
                 sparse_threshold: float = 0.5,
                 nonzero_ratio_min: float = 0.3):
        """
        Args:
            weights: Custom factor weights (default: domain prior)
            sparse_threshold: Avg claims/month threshold for filtering
            nonzero_ratio_min: Minimum ratio of non-zero values
        """
        # 컬럼명 표준화는 실제 데이터 처리 함수에서 적용
        # Default domain prior weights
        self.default_weights = {
            'ratio': 0.20,
            'conf': 0.15,
            'season': 0.30,
            'ampl': 0.15,
            'inflect': 0.20
        }
        
        self.weights = weights if weights else self.default_weights.copy()
        self.sparse_threshold = sparse_threshold
        self.nonzero_ratio_min = nonzero_ratio_min
        
        # Normalization stats (computed during fit)
        self.norm_stats = {}
        
    def _normalize_factor(self, values: np.ndarray, factor_name: str) -> np.ndarray:
        """Min-max normalization with NaN handling"""
        valid_mask = ~np.isnan(values)
        if not valid_mask.any():
            return np.zeros_like(values)
        
        valid_values = values[valid_mask]
        min_val = valid_values.min()
        max_val = valid_values.max()
        
        # Store stats for later use
        self.norm_stats[factor_name] = {'min': min_val, 'max': max_val}
        
        if max_val - min_val < 1e-10:
            return np.zeros_like(values)
        
        normalized = np.zeros_like(values)
        normalized[valid_mask] = (values[valid_mask] - min_val) / (max_val - min_val)
        return normalized
    
    def calculate_f1_growth_ratio(self, 
                                   forecast_values: np.ndarray,
                                   historical_data: np.ndarray) -> float:
        """
        F1: Growth Ratio = mean(ŷ[t+1:t+h]) / mean(y[t-12:t-1])
        
        Args:
            forecast_values: Future predictions (H months)
            historical_data: Past observations (at least 12 months)
        
        Returns:
            Growth ratio (>1 indicates increase)
        """
        if len(historical_data) < 12:
            return np.nan
        
        recent_12m = historical_data[-12:]
        mean_recent = recent_12m.mean()
        
        if mean_recent < 0.01:
            return np.nan
        
        mean_forecast = forecast_values.mean()
        ratio = mean_forecast / mean_recent
        
        return float(ratio)
    
    def calculate_f2_confidence(self,
                                mape: float,
                                pi_lower: Optional[np.ndarray] = None,
                                pi_upper: Optional[np.ndarray] = None,
                                actual: Optional[np.ndarray] = None) -> float:
        """
        F2: Confidence = 0.5·π_compression + 0.5·coverage_80
        
        Args:
            mape: Mean Absolute Percentage Error
            pi_lower: Prediction interval lower bound (optional)
            pi_upper: Prediction interval upper bound (optional)
            actual: Actual values for coverage calculation (optional)
        
        Returns:
            Confidence score [0, 1]
        """
        # If PI data available, use hybrid metric
        if pi_lower is not None and pi_upper is not None:
            # π_compression = 1 - mean(PI_width) / mean(point_forecast)
            pi_width = pi_upper - pi_lower
            mean_width = pi_width.mean()
            mean_forecast = ((pi_upper + pi_lower) / 2).mean()
            
            compression = 1 - (mean_width / mean_forecast) if mean_forecast > 0 else 0
            compression = np.clip(compression, 0, 1)
            
            # Coverage: proportion of actuals within 80% PI
            if actual is not None and len(actual) == len(pi_lower):
                within_pi = ((actual >= pi_lower) & (actual <= pi_upper)).mean()
                coverage = within_pi
            else:
                coverage = 0.8  # Assume nominal coverage
            
            confidence = 0.5 * compression + 0.5 * coverage
        else:
            # Fallback: MAPE-based confidence
            if pd.isna(mape) or mape > 1000:
                confidence = 0.0
            else:
                # Transform MAPE to [0,1] confidence
                # MAPE=0 → conf=1, MAPE=100 → conf=0.5, MAPE≥200 → conf=0
                confidence = max(0, 1 - mape / 200)
        
        return float(np.clip(confidence, 0, 1))
    
    def calculate_f3_seasonality(self, 
                                  historical_data: np.ndarray,
                                  period: int = 12) -> float:
        """
        F3: Seasonality Strength = 1 - Var(resid) / Var(y)
        Using STL decomposition with fallback for sparse data
        
        Args:
            historical_data: Time series data (≥2 seasons recommended)
            period: Seasonal period (12 for monthly)
        
        Returns:
            Seasonality strength [0, 1]
        """
        if len(historical_data) < 2 * period:
            return np.nan
        
        try:
            # STL decomposition
            stl = STL(historical_data, seasonal=period + 1, robust=True)
            result = stl.fit()
            
            seasonal = result.seasonal
            resid = result.resid
            
            var_resid = np.var(resid)
            var_y = np.var(historical_data)
            
            if var_y < 1e-10:
                return 0.0
            
            strength = 1 - (var_resid / var_y)
            strength = max(0, strength)  # Ensure non-negative
            
            return float(strength)
        except:
            # Fallback: Month-to-month autocorrelation (lag=12)
            # For sparse data, use simple seasonal pattern detection
            try:
                if len(historical_data) >= 24:
                    # Calculate correlation between same months across years
                    correlations = []
                    n_years = len(historical_data) // period
                    
                    for month in range(period):
                        values = [historical_data[year * period + month] 
                                 for year in range(n_years) 
                                 if year * period + month < len(historical_data)]
                        
                        if len(values) >= 2 and np.std(values) > 0:
                            # Measure consistency across years
                            cv = np.std(values) / (np.mean(values) + 0.01)
                            consistency = max(0, 1 - cv)
                            correlations.append(consistency)
                    
                    if correlations:
                        # Average consistency across months
                        return float(np.mean(correlations))
                
                return 0.0
            except:
                return 0.0
    
    # calculate_f4_amplitude는 compute_amplitude로 대체됨
    
    def calculate_f5_inflection(self,
                                 historical_data: np.ndarray,
                                 recent_window: int = 6) -> float:
        """
        F5: Rising-Inflection Risk = 0.5·norm(accel) + 0.5·cp_prob
        
        Components:
        - accel: Recent acceleration (2nd derivative > 0)
        - cp_prob: Changepoint probability (ruptures PELT)
        
        Args:
            historical_data: Time series data
            recent_window: Window for acceleration calculation
        
        Returns:
            Inflection risk score [0, 1]
        """
        if len(historical_data) < recent_window + 2:
            return np.nan
        
        # Component 1: Acceleration
        recent = historical_data[-recent_window:]
        
        # 1st derivative (velocity)
        velocity = np.diff(recent)
        
        # 2nd derivative (acceleration)
        if len(velocity) > 1:
            acceleration = np.diff(velocity)
            
            # Positive acceleration average
            pos_accel = acceleration[acceleration > 0]
            if len(pos_accel) > 0 and velocity.std() > 0:
                accel_score = pos_accel.mean() / velocity.std()
                accel_score = np.clip(accel_score, 0, 3) / 3  # Normalize to [0,1]
            else:
                accel_score = 0.0
        else:
            accel_score = 0.0
        
        # Component 2: Changepoint probability
        try:
            # Use PELT algorithm for changepoint detection
            algo = rpt.Pelt(model="rbf", min_size=3, jump=1)
            algo.fit(historical_data)
            
            # Detect changepoints (penalty controls sensitivity)
            penalty = np.log(len(historical_data)) * historical_data.var()
            changepoints = algo.predict(pen=penalty)
            
            # Recent changepoint probability
            if len(changepoints) > 1:
                recent_cp = [cp for cp in changepoints if cp >= len(historical_data) - recent_window]
                cp_prob = len(recent_cp) / max(1, len(changepoints) - 1)  # Exclude endpoint
            else:
                cp_prob = 0.0
        except:
            cp_prob = 0.0
        
        # Combined inflection risk
        inflection = 0.5 * accel_score + 0.5 * cp_prob
        
        return float(inflection)
    
    def compute_5factors(self,
                         series_id: str,
                         forecast_values: np.ndarray,
                         historical_data: np.ndarray,
                         mape: float = None,
                         pi_lower: np.ndarray = None,
                         pi_upper: np.ndarray = None) -> Dict:
        """
        Compute all 5 factors for a series
        
        Returns:
            {
                'series_id': str,
                'f1_ratio': float,
                'f2_conf': float,
                'f3_season': float,
                'f4_ampl': float,
                'f5_inflect': float,
                'sparse_flag': bool,
                'sparse_reason': str
            }
        """
        # Sparse series detection
        mean_y = historical_data.mean()
        nonzero_ratio = np.count_nonzero(historical_data) / len(historical_data)
        
        sparse_flag = (mean_y < self.sparse_threshold) or (nonzero_ratio < self.nonzero_ratio_min)
        
        if sparse_flag:
            reason = []
            if mean_y < self.sparse_threshold:
                reason.append(f'avg={mean_y:.2f}<{self.sparse_threshold}')
            if nonzero_ratio < self.nonzero_ratio_min:
                reason.append(f'nonzero={nonzero_ratio:.1%}<{self.nonzero_ratio_min:.1%}')
            sparse_reason = '; '.join(reason)
        else:
            sparse_reason = ''
        
        # Compute factors
        f1 = self.calculate_f1_growth_ratio(forecast_values, historical_data)
        f2 = self.calculate_f2_confidence(mape, pi_lower, pi_upper)
        f3 = self.calculate_f3_seasonality(historical_data)
        # compute_amplitude is the implemented method for F4 (amplitude)
        f4 = self.compute_amplitude(historical_data)
        f5 = self.calculate_f5_inflection(historical_data)
        
        return {
            'series_id': series_id,
            'f1_ratio': f1,
            'f2_conf': f2,
            'f3_season': f3,
            'f4_ampl': f4,
            'f5_inflect': f5,
            'sparse_flag': sparse_flag,
            'sparse_reason': sparse_reason,
            'mean_y': mean_y,
            'nonzero_ratio': nonzero_ratio
        }
    
    def compute_ews_score(self, factors: Dict) -> float:
        """
        Compute weighted EWS score from 5 factors
        
        Args:
            factors: Output from compute_5factors
        
        Returns:
            EWS score [0, 1] (normalized)
        """
        if factors['sparse_flag']:
            return 0.0
        
        # Extract factor values
        f_values = np.array([
            factors['f1_ratio'],
            factors['f2_conf'],
            factors['f3_season'],
            factors['f4_ampl'],
            factors['f5_inflect']
        ])
        
        # Handle NaN
        valid_mask = ~np.isnan(f_values)
        if not valid_mask.any():
            return 0.0
        
        # Apply weights (only to valid factors)
        w_array = np.array([
            self.weights['ratio'],
            self.weights['conf'],
            self.weights['season'],
            self.weights['ampl'],
            self.weights['inflect']
        ])
        
        # Normalize factors to [0,1]
        f_normalized = f_values.copy()
        for i, fname in enumerate(['ratio', 'conf', 'season', 'ampl', 'inflect']):
            if valid_mask[i]:
                if fname in self.norm_stats:
                    stats = self.norm_stats[fname]
                    f_normalized[i] = (f_values[i] - stats['min']) / max(stats['max'] - stats['min'], 1e-10)
                # conf is already [0,1], others need normalization
        
        # Weighted sum (renormalize weights for valid factors only)
        valid_weights = w_array[valid_mask]
        valid_weights = valid_weights / valid_weights.sum()
        
        ews = np.sum(f_normalized[valid_mask] * valid_weights)
        
        return float(np.clip(ews, 0, 1))
    
    def learn_weights(self,
                      factors_df: pd.DataFrame,
                      labels: np.ndarray,
                      delta: float = 0.3,
                      horizon: int = 6) -> Dict:
        """
        Learn optimal weights using Logistic Regression
        
        Args:
            factors_df: DataFrame with f1-f5 columns
            labels: Binary labels (1=alarm positive, 0=negative)
            delta: Growth threshold for positive label
            horizon: Forecast horizon
        
        Returns:
            {
                'weights': dict,
                'f1_score': float,
                'pr_auc': float,
                'threshold': float
            }
        """
        # Prepare feature matrix
        feature_cols = ['f1_ratio', 'f2_conf', 'f3_season', 'f4_ampl', 'f5_inflect']
        X = factors_df[feature_cols].values
        
        # Remove NaN rows
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        y_valid = labels[valid_mask]
        
        if len(X_valid) < 10:
            print("[WARNING] Not enough valid samples for weight learning")
            return {
                'weights': self.default_weights,
                'f1_score': 0.0,
                'pr_auc': 0.0,
                'threshold': 0.5
            }
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # Logistic regression with L2 (positive coefficients preferred)
        clf = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        clf.fit(X_scaled, y_valid)
        
        # Extract coefficients
        coef = clf.coef_[0]
        
        # Make positive and normalize
        coef_positive = np.abs(coef)
        coef_normalized = coef_positive / coef_positive.sum()
        
        learned_weights = {
            'ratio': float(coef_normalized[0]),
            'conf': float(coef_normalized[1]),
            'season': float(coef_normalized[2]),
            'ampl': float(coef_normalized[3]),
            'inflect': float(coef_normalized[4])
        }
        
        # Evaluate
        y_pred_proba = clf.predict_proba(X_scaled)[:, 1]
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_valid, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        
        # PR-AUC
        pr_auc_score = auc(recall, precision)
        
        return {
            'weights': learned_weights,
            'f1_score': float(best_f1),
            'pr_auc': float(pr_auc_score),
            'threshold': float(best_threshold)
        }


def generate_labels(df_actual: pd.DataFrame,
                    df_forecast: pd.DataFrame,
                    horizon: int = 6,
                    delta: float = 0.3) -> np.ndarray:
    """
    Generate binary labels for backtest
    
    Positive: Σ(actual[t+1:t+H]) ≥ (1+δ) × H × mean(actual[t-12:t-1])
    
    Args:
        df_actual: Actual data with columns [series_id, year, month, claim_count]
        df_forecast: Forecast data (for series list)
        horizon: Forecast horizon (months)
        delta: Growth threshold (0.3 = 30% increase)
    
    Returns:
        Binary labels array (aligned with df_forecast series order)
    """
    labels = []
    
    for series_id in df_forecast['series_id'].unique():
        series_data = df_actual[df_actual['series_id'] == series_id].sort_values(['year', 'month'])
        
        if len(series_data) < 12 + horizon:
            labels.append(0)
            continue
        
        # Recent 12 months
        recent_12 = series_data.iloc[-(12+horizon):-horizon]['claim_count'].values
        mean_recent = recent_12.mean()
        
        # Future H months
        future_h = series_data.iloc[-horizon:]['claim_count'].values
        sum_future = future_h.sum()
        
        # Threshold
        threshold = (1 + delta) * horizon * mean_recent
        
        label = 1 if sum_future >= threshold else 0
        labels.append(label)
    
    return np.array(labels)


def save_threshold_config(output_path: str,
                          weights: Dict,
                          cutoff_h3: float,
                          cutoff_h6: float,
                          f1_score: float,
                          pr_auc: float,
                          cv_scheme: str = "rolling3fold_24m"):
    """Save threshold.json"""
    config = {
        'weights': weights,
        'cutoff': {
            'H3': round(cutoff_h3, 2),
            'H6': round(cutoff_h6, 2)
        },
        'metric': {
            'F1': round(f1_score, 2),
            'PRAUC': round(pr_auc, 2)
        },
        'cv': cv_scheme,
        'updated': datetime.now().strftime('%Y-%m-%d')
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"[SUCCESS] Threshold config saved: {output_path}")


def generate_ews_report(
    forecast_parquet_path: str,
    json_dir: str,
    metadata_path: str,
    output_path: str,
    threshold_path: str = None,
    top_n: int = 10,
    learn_weights: bool = False
) -> pd.DataFrame:
    """
    Generate comprehensive EWS report with 5-factor scoring
    
    Args:
        forecast_parquet_path: Forecast data path
        json_dir: Historical data JSON directory
        metadata_path: training_results.csv path
        output_path: Output CSV path for ews_scores.csv
        threshold_path: threshold.json path (for weight learning)
        top_n: Number of top series to display
        learn_weights: Whether to perform weight learning
    
    Returns:
        EWS scores DataFrame
    """
    # Initialize scorer
    scorer = EWSScorer()

    # Load data
    df_forecast = pd.read_parquet(forecast_parquet_path)
    df_metadata = pd.read_csv(metadata_path)

    # Compute 5-factors for all series, using latest month for forecast window
    all_factors = []
    candidate_details = []

    for series_id in df_forecast['series_id'].unique():
        # Load JSON
        safe_name = series_id.replace('/', '_').replace('|', '_').replace('\\', '_')
        json_path = Path(json_dir) / f"{safe_name}.json"

        if not json_path.exists():
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df_hist = pd.DataFrame(data['data'])
        # Use all available data for historical
        historical_data = df_hist['claim_count'].values

        if len(historical_data) < 12:
            continue

        # Get forecast for next 6 months from latest available month
        # Find latest year/month in df_hist
        latest_row = df_hist.sort_values(['year', 'month']).iloc[-1]
        latest_year, latest_month = int(latest_row['year']), int(latest_row['month'])

        # Filter forecast for next 6 months
        series_forecast = df_forecast[df_forecast['series_id'] == series_id].copy()
        # Support both 'year', 'month' columns
        if 'year' in series_forecast.columns and 'month' in series_forecast.columns:
            # Compute yyyymm for forecast
            series_forecast['yyyymm'] = series_forecast['year'] * 100 + series_forecast['month']
            start_yyyymm = latest_year * 100 + latest_month + 1
            forecast_window = [((latest_year + (latest_month + i) // 12), ((latest_month + i) % 12) or 12) for i in range(1, 7)]
            forecast_mask = series_forecast.apply(lambda r: (r['year'], r['month']) in forecast_window, axis=1)
            if 'forecast_value' in series_forecast.columns:
                forecast_values = series_forecast[forecast_mask]['forecast_value'].values
            elif 'y_pred' in series_forecast.columns:
                forecast_values = series_forecast[forecast_mask]['y_pred'].values
            else:
                forecast_values = np.zeros(6)
            forecast_months = series_forecast[forecast_mask][['year', 'month']].values.tolist()
        else:
            # Fallback: just take first 6 forecast values
            if 'forecast_value' in series_forecast.columns:
                forecast_values = series_forecast['forecast_value'].values[:6]
            elif 'y_pred' in series_forecast.columns:
                forecast_values = series_forecast['y_pred'].values[:6]
            elif 'predicted_value' in series_forecast.columns:
                forecast_values = series_forecast['predicted_value'].values[:6]
            else:
                forecast_values = np.zeros(6)
            forecast_months = []

        if len(forecast_values) < 6:
            continue

        # Get WMAPE from metadata
        model_info = df_metadata[df_metadata['series_id'] == series_id]
        mape = None
        if len(model_info) > 0:
            if 'wmape' in model_info.columns:
                mape = model_info.iloc[0]['wmape']
            elif 'mape' in model_info.columns:
                mape = model_info.iloc[0]['mape']

        # Compute factors
        factors = scorer.compute_5factors(
            series_id=series_id,
            forecast_values=forecast_values,
            historical_data=historical_data,
            mape=mape
        )

        # Attach forecast window info for candidates
        factors['forecast_months'] = forecast_months
        factors['forecast_values'] = forecast_values.tolist()
        all_factors.append(factors)

    df_factors = pd.DataFrame(all_factors)
    # KeyError 방지: 'sparse_flag' 컬럼이 없거나 empty일 때 빈 결과 반환
    output_cols = ['rank', 'series_id', 'ews_score', 'level', 'f1_ratio', 'f2_conf', 
                   'f3_season', 'f4_ampl', 'f5_inflect', 'candidate', 'rationale']
    if df_factors.empty or 'sparse_flag' not in df_factors.columns:
        df_output = pd.DataFrame(columns=output_cols)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n[SUCCESS] EWS report saved: {output_path}")
        return df_output
    # Filter sparse series
    df_valid = df_factors[~df_factors['sparse_flag']].copy()
    # Normalize factors for EWS calculation
    for fname in ['f1_ratio', 'f3_season', 'f4_ampl', 'f5_inflect']:
        if fname in df_valid.columns:
            scorer._normalize_factor(df_valid[fname].values, fname)
    # Compute EWS scores
    df_valid['ews_score'] = df_valid.apply(lambda row: scorer.compute_ews_score(row.to_dict()), axis=1)
    # Apply candidacy filters: S≥0.4 and A≥0.3
    df_valid['candidate'] = (df_valid['f3_season'] >= 0.4) & (df_valid['f4_ampl'] >= 0.3)
    df_valid['low_confidence'] = df_valid['f2_conf'] < 0.2
    # Determine risk level
    def get_level(score, conf):
        if conf < 0.2:
            return 'LOW_CONF'
        elif score >= 0.7:
            return 'HIGH'
        elif score >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    df_valid['level'] = df_valid.apply(lambda r: get_level(r['ews_score'], r['f2_conf']), axis=1)
    # Generate rationale
    def make_rationale(row):
        parts = []
        if row['f1_ratio'] > 1.5:
            parts.append(f"증가율{row['f1_ratio']:.1f}x")
        if row['f3_season'] >= 0.6:
            parts.append(f"강한계절성{row['f3_season']:.2f}")
        if row['f4_ampl'] >= 0.5:
            parts.append(f"큰진폭{row['f4_ampl']:.2f}")
        if row['f5_inflect'] >= 0.5:
            parts.append(f"변곡위험{row['f5_inflect']:.2f}")
        if row['f2_conf'] < 0.2:
            parts.append(f"낮은신뢰도{row['f2_conf']:.2f}")
        return '; '.join(parts) if parts else 'normal'
    df_valid['rationale'] = df_valid.apply(make_rationale, axis=1)
    # Only keep candidate series
    df_candidates = df_valid[df_valid['candidate']].copy()
    df_candidates = df_candidates.sort_values('ews_score', ascending=False)
    df_candidates['rank'] = range(1, len(df_candidates) + 1)
    # Save
    output_cols = ['rank', 'series_id', 'ews_score', 'level', 'f1_ratio', 'f2_conf', 
                   'f3_season', 'f4_ampl', 'f5_inflect', 'candidate', 'rationale', 'forecast_months', 'forecast_values']
    df_output = df_candidates[output_cols].copy()
    if 'forecast_months' in df_output.columns:
        df_output['forecast_months'] = df_output['forecast_months'].apply(json.dumps)
    if 'forecast_values' in df_output.columns:
        df_output['forecast_values'] = df_output['forecast_values'].apply(json.dumps)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n{'='*80}")
    print(f"EWS 5-FACTOR SCORING REPORT (CANDIDATES ONLY)")
    print(f"{'='*80}")
    print(f"Total series processed: {len(df_factors)}")
    print(f"Sparse filtered: {df_factors['sparse_flag'].sum()}")
    print(f"Valid candidates (S≥0.4, A≥0.3): {df_candidates['candidate'].sum()}")
    print(f"\nWeights: {scorer.weights}")
    print(f"\n{'='*80}")
    print(f"TOP {top_n} HIGH-RISK SERIES")
    print(f"{'='*80}")
    for _, row in df_output.head(top_n).iterrows():
        print(f"\n[{int(row['rank'])}] {row['series_id']}")
        print(f"  EWS Score: {row['ews_score']:.3f} ({row['level']})")
        print(f"  F1(증가율): {row['f1_ratio']:.2f}x")
        print(f"  F2(신뢰도): {row['f2_conf']:.2f}")
        print(f"  F3(계절성): {row['f3_season']:.2f}")
        print(f"  F4(진폭): {row['f4_ampl']:.2f}")
        print(f"  F5(변곡): {row['f5_inflect']:.2f}")
        print(f"  근거: {row['rationale']}")
        if row['forecast_months']:
            print(f"  예측월: {row['forecast_months']}")
            print(f"  예측값: {row['forecast_values']}")
    print(f"\n[SUCCESS] EWS report saved: {output_path}")
    return df_output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate 5-Factor EWS Scores")
    parser.add_argument("--forecast", type=str, required=True,
                        help="Forecast parquet file path")
    parser.add_argument("--json-dir", type=str, default="data/features",
                        help="JSON data directory")
    parser.add_argument("--metadata", type=str,
                        default="artifacts/models/base_monthly/training_results.csv",
                        help="Model metadata CSV path")
    parser.add_argument("--output", type=str,
                        default="artifacts/metrics/ews_scores.csv",
                        help="Output CSV path")
    parser.add_argument("--threshold", type=str,
                        default="artifacts/metrics/threshold.json",
                        help="Threshold config path")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top series to display")
    parser.add_argument("--learn-weights", action="store_true",
                        help="Perform weight learning (requires labels)")
    
    args = parser.parse_args()
    
    df_ews = generate_ews_report(
        forecast_parquet_path=args.forecast,
        json_dir=args.json_dir,
        metadata_path=args.metadata,
        output_path=args.output,
        threshold_path=args.threshold,
        top_n=args.top_n,
        learn_weights=args.learn_weights
    )
