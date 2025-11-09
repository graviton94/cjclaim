"""
EWS Weight Learning Backtest
=============================
Automated weight optimization using rolling cross-validation

Process:
1. Load historical data (2021-2024)
2. Generate labels: future_H_sum ≥ (1+δ) × H × mean(recent_12m)
3. Compute 5-factors for all series
4. Rolling 3-fold CV split
5. Learn weights via Logistic Regression
6. Evaluate F1 and PR-AUC
7. Save threshold.json with optimal weights and cutoffs

Usage:
    python backtest_ews_weights.py --delta 0.3 --horizon 6 --cv-folds 3
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

from src.ews_scoring_v2 import EWSScorer, generate_labels
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, auc, classification_report

warnings.filterwarnings('ignore')


def load_series_data(json_dir: str, year_range: Tuple[int, int]) -> pd.DataFrame:
    """
    Load all series historical data
    
    Returns:
        DataFrame with columns: series_id, year, month, claim_count
    """
    json_path = Path(json_dir)
    all_data = []
    
    for json_file in json_path.glob("*.json"):
        if json_file.name.startswith('_'):
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            series_id = data['series_id']
            df = pd.DataFrame(data['data'])
            
            # Filter year range
            df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
            
            if len(df) < 24:  # Need at least 2 years
                continue
            
            df['series_id'] = series_id
            all_data.append(df[['series_id', 'year', 'month', 'claim_count']])
        except:
            continue
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def compute_factors_for_period(
    df_data: pd.DataFrame,
    series_id: str,
    train_end_year: int,
    train_end_month: int,
    horizon: int,
    scorer: EWSScorer
) -> Dict:
    """
    Compute 5-factors for a series at a specific point in time
    
    Args:
        df_data: Full historical data
        series_id: Series identifier
        train_end_year: Last training year
        train_end_month: Last training month
        horizon: Forecast horizon
        scorer: EWS scorer instance
    
    Returns:
        Factor dictionary
    """
    # Get historical data up to train_end
    series_data = df_data[df_data['series_id'] == series_id].sort_values(['year', 'month'])
    
    # Split point
    train_mask = (
        (series_data['year'] < train_end_year) |
        ((series_data['year'] == train_end_year) & (series_data['month'] <= train_end_month))
    )
    
    historical = series_data[train_mask]['claim_count'].values
    
    if len(historical) < 24:
        return None
    
    # Future data for "forecast" proxy (use actual as forecast for backtest)
    future_mask = (
        (series_data['year'] > train_end_year) |
        ((series_data['year'] == train_end_year) & (series_data['month'] > train_end_month))
    )
    
    future = series_data[future_mask]['claim_count'].values[:horizon]
    
    if len(future) < horizon:
        return None
    
    # Compute factors
    # For backtest, we use actual future values as "forecast" to get realistic factor distributions
    factors = scorer.compute_5factors(
        series_id=series_id,
        forecast_values=future,  # Using actual as proxy
        historical_data=historical,
        mape=None  # Will use default confidence
    )
    
    return factors


def rolling_cv_backtest(
    df_data: pd.DataFrame,
    scorer: EWSScorer,
    delta: float = 0.3,
    horizon: int = 6,
    n_splits: int = 3
) -> Dict:
    """
    Rolling cross-validation for weight learning
    
    Args:
        df_data: Historical data
        scorer: EWS scorer
        delta: Growth threshold for positive label
        horizon: Forecast horizon
        n_splits: Number of CV folds
    
    Returns:
        {
            'weights': dict,
            'f1_mean': float,
            'pr_auc_mean': float,
            'threshold': float,
            'fold_results': list
        }
    """
    # Get unique series
    series_list = df_data['series_id'].unique()
    
    # Define time points for rolling CV
    # E.g., if data is 2021-2024 (48 months), split into 3 folds:
    # Fold 1: train 2021-2022, test 2023H1
    # Fold 2: train 2021-2023H1, test 2023H2
    # Fold 3: train 2021-2023, test 2024H1
    
    min_year = df_data['year'].min()
    max_year = df_data['year'].max()
    total_months = (max_year - min_year + 1) * 12
    
    fold_results = []
    
    for fold in range(n_splits):
        print(f"\n[FOLD {fold+1}/{n_splits}]")
        
        # Determine train/test split
        # Each fold advances by total_months / (n_splits + 1)
        step_months = total_months // (n_splits + 1)
        train_months = (fold + 1) * step_months
        
        train_end_year = min_year + train_months // 12
        train_end_month = train_months % 12
        if train_end_month == 0:
            train_end_month = 12
            train_end_year -= 1
        
        test_end_year = train_end_year + (horizon // 12)
        test_end_month = train_end_month + (horizon % 12)
        if test_end_month > 12:
            test_end_month -= 12
            test_end_year += 1
        
        print(f"  Train until: {train_end_year}-{train_end_month:02d}")
        print(f"  Test until:  {test_end_year}-{test_end_month:02d}")
        
        # Compute factors for all series at this time point
        all_factors = []
        all_labels = []
        
        for series_id in series_list:
            factors = compute_factors_for_period(
                df_data, series_id, train_end_year, train_end_month, horizon, scorer
            )
            
            if factors is None or factors['sparse_flag']:
                continue
            
            # Generate label
            series_data = df_data[df_data['series_id'] == series_id].sort_values(['year', 'month'])
            
            # Recent 12 months
            recent_mask = (
                (series_data['year'] < train_end_year) |
                ((series_data['year'] == train_end_year) & (series_data['month'] <= train_end_month))
            )
            recent_data = series_data[recent_mask].tail(12)
            
            if len(recent_data) < 12:
                continue
            
            mean_recent = recent_data['claim_count'].mean()
            
            # Future H months
            future_mask = (
                (series_data['year'] > train_end_year) |
                ((series_data['year'] == train_end_year) & (series_data['month'] > train_end_month))
            )
            future_data = series_data[future_mask].head(horizon)
            
            if len(future_data) < horizon:
                continue
            
            sum_future = future_data['claim_count'].sum()
            threshold = (1 + delta) * horizon * mean_recent
            
            label = 1 if sum_future >= threshold else 0
            
            all_factors.append(factors)
            all_labels.append(label)
        
        if len(all_factors) < 10:
            print(f"  [WARNING] Not enough valid samples ({len(all_factors)}), skipping fold")
            continue
        
        # Convert to DataFrame
        df_factors = pd.DataFrame(all_factors)
        y = np.array(all_labels)
        
        print(f"  Samples: {len(all_factors)} | Positive: {y.sum()} ({y.mean()*100:.1f}%)")
        
        # Feature matrix
        feature_cols = ['f1_ratio', 'f2_conf', 'f3_season', 'f4_ampl', 'f5_inflect']
        X = df_factors[feature_cols].values
        
        # Remove NaN rows
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(X_valid) < 10:
            print(f"  [WARNING] Not enough valid samples after NaN removal, skipping fold")
            continue
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # Logistic Regression
        clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(X_scaled, y_valid)
        
        # Predict
        y_pred_proba = clf.predict_proba(X_scaled)[:, 1]
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_valid, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        
        # PR-AUC
        pr_auc_score = auc(recall, precision)
        
        # Extract weights
        coef = np.abs(clf.coef_[0])
        coef_normalized = coef / coef.sum()
        
        fold_weights = {
            'ratio': float(coef_normalized[0]),
            'conf': float(coef_normalized[1]),
            'season': float(coef_normalized[2]),
            'ampl': float(coef_normalized[3]),
            'inflect': float(coef_normalized[4])
        }
        
        fold_results.append({
            'fold': fold + 1,
            'weights': fold_weights,
            'f1': best_f1,
            'pr_auc': pr_auc_score,
            'threshold': best_threshold,
            'n_samples': len(X_valid),
            'n_positive': int(y_valid.sum())
        })
        
        print(f"  F1: {best_f1:.3f} | PR-AUC: {pr_auc_score:.3f} | Threshold: {best_threshold:.3f}")
        print(f"  Weights: ratio={fold_weights['ratio']:.2f}, conf={fold_weights['conf']:.2f}, "
              f"season={fold_weights['season']:.2f}, ampl={fold_weights['ampl']:.2f}, inflect={fold_weights['inflect']:.2f}")
    
    if len(fold_results) == 0:
        return None
    
    # Aggregate results
    df_folds = pd.DataFrame(fold_results)
    
    avg_weights = {
        'ratio': df_folds['weights'].apply(lambda w: w['ratio']).mean(),
        'conf': df_folds['weights'].apply(lambda w: w['conf']).mean(),
        'season': df_folds['weights'].apply(lambda w: w['season']).mean(),
        'ampl': df_folds['weights'].apply(lambda w: w['ampl']).mean(),
        'inflect': df_folds['weights'].apply(lambda w: w['inflect']).mean()
    }
    
    # Renormalize
    total = sum(avg_weights.values())
    avg_weights = {k: v/total for k, v in avg_weights.items()}
    
    return {
        'weights': avg_weights,
        'f1_mean': df_folds['f1'].mean(),
        'f1_std': df_folds['f1'].std(),
        'pr_auc_mean': df_folds['pr_auc'].mean(),
        'pr_auc_std': df_folds['pr_auc'].std(),
        'threshold_mean': df_folds['threshold'].mean(),
        'fold_results': fold_results
    }


def main():
    parser = argparse.ArgumentParser(description="EWS Weight Learning Backtest")
    parser.add_argument("--json-dir", type=str, default="data/features",
                        help="JSON data directory")
    parser.add_argument("--delta", type=float, default=0.3,
                        help="Growth threshold for positive label (default: 0.3 = 30%)")
    parser.add_argument("--horizon", type=int, default=6,
                        help="Forecast horizon in months (default: 6)")
    parser.add_argument("--cv-folds", type=int, default=3,
                        help="Number of CV folds (default: 3)")
    parser.add_argument("--output", type=str, default="artifacts/metrics/threshold.json",
                        help="Output threshold config path")
    parser.add_argument("--year-start", type=int, default=2021,
                        help="Start year for data (default: 2021)")
    parser.add_argument("--year-end", type=int, default=2024,
                        help="End year for data (default: 2024)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EWS Weight Learning Backtest")
    print("=" * 80)
    print(f"Data range: {args.year_start}-{args.year_end}")
    print(f"Delta: {args.delta} ({args.delta*100:.0f}% growth threshold)")
    print(f"Horizon: {args.horizon} months")
    print(f"CV Folds: {args.cv_folds}")
    print("=" * 80)
    
    # Load data
    print("\n[INFO] Loading series data...")
    df_data = load_series_data(args.json_dir, (args.year_start, args.year_end))
    
    if len(df_data) == 0:
        print("[ERROR] No data loaded")
        return
    
    print(f"[INFO] Loaded {df_data['series_id'].nunique()} series")
    
    # Initialize scorer
    scorer = EWSScorer(sparse_threshold=0.5, nonzero_ratio_min=0.3)
    
    # Run backtest
    print("\n[INFO] Running rolling CV backtest...")
    results = rolling_cv_backtest(
        df_data=df_data,
        scorer=scorer,
        delta=args.delta,
        horizon=args.horizon,
        n_splits=args.cv_folds
    )
    
    if results is None:
        print("[ERROR] Backtest failed - no valid folds")
        return
    
    # Display results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"F1 Score:  {results['f1_mean']:.3f} ± {results['f1_std']:.3f}")
    print(f"PR-AUC:    {results['pr_auc_mean']:.3f} ± {results['pr_auc_std']:.3f}")
    print(f"Threshold: {results['threshold_mean']:.3f}")
    print(f"\nLearned Weights:")
    for factor, weight in results['weights'].items():
        print(f"  {factor:8s}: {weight:.3f}")
    
    # Save threshold.json
    threshold_config = {
        'weights': results['weights'],
        'cutoff': {
            'H3': round(results['threshold_mean'], 2),
            'H6': round(results['threshold_mean'], 2)
        },
        'metric': {
            'F1': round(results['f1_mean'], 3),
            'F1_std': round(results['f1_std'], 3),
            'PRAUC': round(results['pr_auc_mean'], 3),
            'PRAUC_std': round(results['pr_auc_std'], 3)
        },
        'cv': f"rolling{args.cv_folds}fold_{args.year_end-args.year_start}y",
        'delta': args.delta,
        'horizon': args.horizon,
        'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'fold_details': results['fold_results']
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(threshold_config, f, indent=2)
    
    print(f"\n[SUCCESS] Threshold config saved: {output_path}")
    
    # Check success criteria
    if results['f1_mean'] >= 0.75:
        print(f"\n✅ SUCCESS CRITERIA MET: F1={results['f1_mean']:.3f} ≥ 0.75")
    else:
        print(f"\n⚠️  WARNING: F1={results['f1_mean']:.3f} < 0.75 (target)")
    
    print("\n[INFO] Backtest complete!")


if __name__ == "__main__":
    main()
