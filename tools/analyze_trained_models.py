import json
import math
from statistics import mean, median

TP = 'artifacts/trained_models.json'
TS = 'artifacts/training_summary.json'

def safe_load(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def fmt(x):
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)

def main():
    trained = safe_load(TP)
    summary = safe_load(TS) if __import__('os').path.exists(TS) else None

    meta = trained.get('metadata', {})
    series = trained.get('series_models', {})
    n_series = len(series)

    print('\nTRAINED MODELS SUMMARY')
    print('----------------------')
    print('Train date:', meta.get('train_date'))
    print('Train until:', f"{meta.get('train_until_year')}-{meta.get('train_until_month')}")
    print('Forecast horizon:', meta.get('forecast_horizon'))
    print('Series in file:', n_series)
    if summary:
        print('Training summary (artifact):')
        for k,v in summary.items():
            print(' ', k, ':', v)

    # Aggregates
    model_counts = {}
    npoints = []
    has_neg_forecast = 0
    has_nan = 0
    huge_forecasts = []
    max_forecast_overall = -math.inf
    seasonal_count = 0
    selection_losses = []

    growth_examples = []  # (series_id, max_yhat, first, last, ratio)

    for sid, info in series.items():
        mt = info.get('model_type', 'unknown')
        model_counts[mt] = model_counts.get(mt, 0) + 1
        n = info.get('n_train_points') or 0
        npoints.append(n)
        spec = info.get('model_spec', {})
        sorder = spec.get('seasonal_order', [0,0,0,0])
        if any(x != 0 for x in sorder):
            seasonal_count += 1

        sel = info.get('selection_loss')
        if sel is not None:
            try:
                selection_losses.append(float(sel))
            except Exception:
                pass

        forecast = info.get('forecast', {}).get('yhat', [])
        if not isinstance(forecast, list):
            # sometimes it's nested or numpy; coerce
            try:
                forecast = list(forecast)
            except Exception:
                forecast = []
        if any((isinstance(x, float) and math.isnan(x)) for x in forecast):
            has_nan += 1
        if any((isinstance(x, (int,float)) and x < 0) for x in forecast):
            has_neg_forecast += 1
        if forecast:
            local_max = max([float(x) for x in forecast])
            if local_max > max_forecast_overall:
                max_forecast_overall = local_max
            if local_max > 1000:
                huge_forecasts.append((sid, local_max))
            first = float(forecast[0])
            last = float(forecast[-1])
            ratio = (last / first) if (first != 0) else (last if last!=0 else 1.0)
            growth_examples.append((sid, local_max, first, last, ratio))

    print('\nOverall counts:')
    for k,v in sorted(model_counts.items(), key=lambda x:-x[1]):
        print(' ', k, ':', v)

    print('\nTraining points per series (n_train_points):')
    if npoints:
        print(' min', min(npoints), 'median', median(npoints), 'mean', round(mean(npoints),2), 'max', max(npoints))

    print('\nSeasonal models count:', seasonal_count)
    print('Series with NaN in forecast:', has_nan)
    print('Series with negative forecast values:', has_neg_forecast)
    print('Max forecast value across all species:', fmt(max_forecast_overall))
    print('Series with extremely large forecasts (>1000):', len(huge_forecasts))

    if selection_losses:
        print('\nSelection loss (sample): min', fmt(min(selection_losses)), 'median', fmt(median(selection_losses)), 'mean', fmt(mean(selection_losses)), 'max', fmt(max(selection_losses)))

    # Top growth examples
    growth_examples.sort(key=lambda x: x[1], reverse=True)
    print('\nTop 10 series by max forecast (sample):')
    for sid, local_max, first, last, ratio in growth_examples[:10]:
        print(' -', sid)
        print('   n_train pts:', series[sid].get('n_train_points'), 'hist_mean:', fmt(series[sid].get('hist_mean')),
              'hist_max:', fmt(series[sid].get('hist_max')))
        print('   forecast yhat (truncated):', [fmt(x) for x in series[sid].get('forecast',{}).get('yhat',[])])

    # Anomaly checks
    anomalies = []
    for sid, local_max, first, last, ratio in growth_examples:
        # flag if ratio > 5 and local_max > 20 or last > 100
        if (ratio > 5 and local_max > 20) or (last > 100):
            anomalies.append((sid, local_max, first, last, ratio))
    print('\nAnomalous series (heuristic):', len(anomalies))
    if anomalies:
        print(' Examples:')
        for sid, local_max, first, last, ratio in anomalies[:5]:
            print('  -', sid, 'max', fmt(local_max), 'first', fmt(first), 'last', fmt(last), 'ratio', fmt(ratio))

    print('\nDone. For deeper checks I can compute AR-root stability for a sample of saved model pickles, or produce CSV diagnostics for failed series.')

if __name__ == '__main__':
    main()
