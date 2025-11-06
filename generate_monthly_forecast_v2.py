# -*- coding: utf-8 -*-
"""
generate_monthly_forecast_v2.py

Generates monthly forecasts for all trained series using parameter-based reconstruction.
Each forecast is saved to artifacts/forecasts/YYYY/forecast_YYYY_MM.parquet

KEY CHANGES (v0.2):
- Horizon: 26 weeks (6 months) - NOT 8 weeks
- NaN checking for confidence intervals with fallback
- Clipping applied: all values ≥ 0
- Required output fields: series_id, year, month, week, yhat, yhat_lower, yhat_upper, 
  horizon_weeks, model_id, order, seasonal_order, generated_at

Usage:
    python generate_monthly_forecast_v2.py --year 2024 --month 1 --horizon 26
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_params(model_path: Path):
    """
    Load stored SARIMAX parameters from pickle file.
    
    Returns:
        dict: Contains 'params', 'order', 'seasonal_order', 'series_id'
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_series_data_from_json(series_id: str, data_dir: Path):
    """
    Load time series data from JSON file.
    
    Args:
        series_id: Series identifier
        data_dir: Directory containing feature JSON files
    
    Returns:
        pd.DataFrame: Series data with columns [year, week, claim_count, week_start_date]
    """
    import json
    
    # Clean series_id for filename (replace invalid characters)
    clean_id = series_id.replace('/', '_').replace('\\', '_').replace(':', '_').replace('|', '_').replace('?', '_').replace('*', '_').replace('<', '_').replace('>', '_').replace('"', '_')
    json_path = data_dir / f"{clean_id}.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        series_data = json.load(f)
    
    data_records = series_data.get('data', [])
    df = pd.DataFrame(data_records)
    
    # Create week_start_date from year and week columns
    # Convert week to string first, then apply zfill
    df['week_str'] = df['week'].astype(int).astype(str).str.zfill(2)
    df['week_start_date'] = pd.to_datetime(df['year'].astype(str) + '-W' + df['week_str'] + '-1', format='%Y-W%W-%w')
    df = df.drop(columns=['week_str'])
    
    return df

def generate_forecast_for_series(series_id: str, model_dir: Path, data_dir: Path,
                                  forecast_month_start: pd.Timestamp, horizon: int = 26):
    """
    Generate forecast for a single series using parameter-based reconstruction.
    
    CRITICAL UPDATES:
    - horizon default: 26 weeks (6 months)
    - NaN checking: if conf_int has NaN, fallback to yhat ± (yhat * 0.2)
    - Clipping: all values ≥ 0
    
    Args:
        series_id: Series identifier
        model_dir: Directory containing trained model pickle files
        data_dir: Directory containing feature JSON files
        forecast_month_start: Start date of forecast month (first day)
        horizon: Number of weeks to forecast (default: 26)
    
    Returns:
        pd.DataFrame: Forecast with columns [series_id, year, month, week, yhat, yhat_lower, yhat_upper,
                      horizon_weeks, model_id, order, seasonal_order, generated_at]
    """
    model_filename = f"{series_id}.pkl"
    model_path = model_dir / model_filename
    
    if not model_path.exists():
        logger.warning(f"Model not found for series {series_id}")
        return None
    
    # Load model parameters
    try:
        model_data = load_model_params(model_path)
        params = model_data['params']
        order = model_data.get('order', (0, 1, 1))
        seasonal_order = model_data.get('seasonal_order', (0, 1, 1, 52))
    except Exception as e:
        logger.error(f"Failed to load model parameters for {series_id}: {e}")
        return None
    
    # Load training data (2021-2023)
    try:
        df = load_series_data_from_json(series_id, data_dir)
        df_train = df[(df['year'] >= 2021) & (df['year'] <= 2023)].copy()
    except Exception as e:
        logger.error(f"Failed to load data for {series_id}: {e}")
        return None
    
    if len(df_train) < 52:
        logger.warning(f"Insufficient data for series {series_id}: {len(df_train)} weeks")
        return None
    
    # Sort by week_start_date
    df_train = df_train.sort_values('week_start_date')
    # Ensure claim_count is a proper array
    if isinstance(df_train['claim_count'].iloc[0], (list, np.ndarray)):
        y = np.array([x[0] if isinstance(x, (list, np.ndarray)) else x for x in df_train['claim_count']])
    else:
        y = df_train['claim_count'].values
    
    # Reconstruct SARIMAX model with stored parameters
    try:
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False)
        
        # Use smooth() to reconstruct state with stored parameters
        result = model.smooth(params)
        
        # Generate forecast
        forecast_result = result.get_forecast(steps=horizon)
        
        # Extract values - handle both pandas Series and numpy arrays
        predicted_mean = forecast_result.predicted_mean
        yhat = predicted_mean.values if hasattr(predicted_mean, 'values') else predicted_mean
        
        conf_int = forecast_result.conf_int(alpha=0.05)  # 95% CI
        
        # Extract confidence interval values
        if hasattr(conf_int, 'iloc'):
            yhat_lower = conf_int.iloc[:, 0].values
            yhat_upper = conf_int.iloc[:, 1].values
        else:
            yhat_lower = conf_int[:, 0]
            yhat_upper = conf_int[:, 1]
        
        # CRITICAL: NaN checking for confidence intervals
        if np.isnan(yhat_lower).any() or np.isnan(yhat_upper).any():
            logger.warning(f"NaN detected in confidence intervals for {series_id}, applying fallback")
            # Fallback: use yhat ± (yhat * 0.2)
            yhat_lower = np.where(np.isnan(yhat_lower), yhat - (yhat * 0.2), yhat_lower)
            yhat_upper = np.where(np.isnan(yhat_upper), yhat + (yhat * 0.2), yhat_upper)
        
        # Apply lower bound clipping (no negative values)
        yhat = np.maximum(yhat, 0)
        yhat_lower = np.maximum(yhat_lower, 0)
        yhat_upper = np.maximum(yhat_upper, 0)
        
        # Ensure coherency: yhat_lower <= yhat <= yhat_upper
        yhat_lower = np.minimum(yhat_lower, yhat)
        yhat_upper = np.maximum(yhat_upper, yhat)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'series_id': series_id,
            'year': forecast_month_start.year,
            'month': forecast_month_start.month,
            'week': range(1, horizon + 1),
            'yhat': yhat,
            'yhat_lower': yhat_lower,
            'yhat_upper': yhat_upper,
            'horizon_weeks': horizon,
            'model_id': model_filename.replace('.pkl', ''),
            'order': str(order),
            'seasonal_order': str(seasonal_order),
            'generated_at': datetime.now()
        })
        
        return forecast_df
        
    except Exception as e:
        logger.error(f"Forecast failed for series {series_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate monthly forecasts for all series')
    parser.add_argument('--year', type=int, required=True, help='Forecast year (e.g., 2024)')
    parser.add_argument('--month', type=int, required=True, help='Forecast month (1-12)')
    parser.add_argument('--horizon', type=int, default=26, help='Forecast horizon in weeks (default: 26)')
    parser.add_argument('--data-dir', type=str, default=None, help='Path to feature JSON directory (optional)')
    parser.add_argument('--models-dir', type=str, default=None, help='Path to models directory (optional)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for forecasts (optional)')
    
    args = parser.parse_args()
    
    # Set paths
    data_dir = Path(args.data_dir) if args.data_dir else Path("data/features")
    models_path = Path(args.models_dir) if args.models_dir else Path("artifacts/models/base_2021_2023")
    output_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/forecasts") / str(args.year)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get forecast month start date
    forecast_month_start = pd.Timestamp(year=args.year, month=args.month, day=1)
    
    # Get all available models
    model_files = list(models_path.glob("*.pkl"))
    logger.info(f"Found {len(model_files)} trained models")
    
    all_forecasts = []
    success_count = 0
    failure_count = 0
    
    # Generate forecasts for each series
    for model_file in model_files:
        series_id = model_file.stem  # Remove .pkl extension
        
        forecast_df = generate_forecast_for_series(
            series_id=series_id,
            model_dir=models_path,
            data_dir=data_dir,
            forecast_month_start=forecast_month_start,
            horizon=args.horizon
        )
        
        if forecast_df is not None:
            all_forecasts.append(forecast_df)
            success_count += 1
            if success_count % 100 == 0:
                logger.info(f"Processed {success_count}/{len(model_files)} series...")
        else:
            failure_count += 1
    
    # Combine all forecasts
    if all_forecasts:
        combined_df = pd.concat(all_forecasts, ignore_index=True)
        
        # Final validation: ensure no NaN or negative values
        nan_check = combined_df[['yhat', 'yhat_lower', 'yhat_upper']].isnull().any().any()
        neg_check = (combined_df[['yhat', 'yhat_lower', 'yhat_upper']] < 0).any().any()
        
        if nan_check:
            logger.error("CRITICAL: NaN values detected in forecast!")
            return 1
        
        if neg_check:
            logger.error("CRITICAL: Negative values detected in forecast!")
            return 1
        
        # Save to parquet
        output_file = output_dir / f"forecast_{args.year}_{args.month:02d}.parquet"
        combined_df.to_parquet(output_file, index=False)
        
        logger.info("=" * 60)
        logger.info(f"✓ Forecasts saved to {output_file}")
        logger.info(f"  Total series: {len(model_files)}")
        logger.info(f"  Success: {success_count}")
        logger.info(f"  Failures: {failure_count}")
        logger.info(f"  Forecast rows: {len(combined_df)}")
        logger.info(f"  Horizon: {args.horizon} weeks")
        logger.info(f"  NaN check: PASS")
        logger.info(f"  Negative value check: PASS")
        logger.info("=" * 60)
    else:
        logger.error("No forecasts generated!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
