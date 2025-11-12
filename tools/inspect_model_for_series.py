import pickle, json, numpy as np, pandas as pd, os, sys
pkl_paths = [
    'artifacts/models/우양냉동(장항)_핫도그_기타.pkl',
    'artifacts/models/base_monthly/우양냉동(장항)_핫도그_기타.pkl',
]
found=None
for p in pkl_paths:
    if os.path.exists(p):
        found=p; break
print('using model:', found)
if not found:
    print('model pkl not found', file=sys.stderr)
    sys.exit(2)
with open(found,'rb') as f:
    obj=pickle.load(f)
print('model type:', type(obj))
# try to get params and summary
try:
    params = getattr(obj, 'params', None)
    print('params sample:', None if params is None else params[:10])
except Exception as e:
    print('params read error:', e)
# try to inspect statsmodels results
try:
    if hasattr(obj, 'model') and hasattr(obj.model, 'endog'):
        endog = obj.model.endog
        print('train endog length', len(endog), 'head:', endog[:10])
    if hasattr(obj, 'arparams'):
        ar = np.r_[1, -obj.arparams]
        roots = np.roots(ar)
        print('AR roots:', roots)
    # try result.get_prediction residual diagnostics if available
    if hasattr(obj, 'fittedvalues'):
        fv = getattr(obj, 'fittedvalues')
        print('fittedvalues head:', fv[:10])
except Exception as e:
    print('inspect error', e)
# load feature json for raw series
feat = 'data/features/우양냉동(장항)_핫도그_기타.json'
if os.path.exists(feat):
    with open(feat,'r',encoding='utf-8') as f:
        jd=json.load(f)
    arr = jd.get('data',[])
    df = pd.DataFrame(arr)
    df = df.sort_values(['year','month'])
    print('\nhistoric sample:\n', df.tail(20).to_string(index=False))
    # show forecast lines from forecasts csv
    fc_csv='artifacts/forecasts/2024/forecast_2024_01.csv'
    if os.path.exists(fc_csv):
        df_fc = pd.read_csv(fc_csv)
        sfc = df_fc[df_fc['series_id']=='우양냉동(장항)_핫도그_기타']
        print('\nforecast rows for series:\n', sfc.to_string(index=False))
else:
    print('feature json missing')
