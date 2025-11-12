"""Summarize differences between trained_models_v3.json and trained_models_v3_rerun.json
and inspect fit logs and failed_series.
Writes artifacts/rerun_summary.json and prints a compact summary.
"""
import json
from pathlib import Path

orig = Path('artifacts/trained_models_v3.json')
rerun = Path('artifacts/trained_models_v3_rerun.json')
fit_log = Path('artifacts/fit_summary_log.json')
failed = Path('artifacts/failed_series.json')
out = Path('artifacts/rerun_summary.json')

if not orig.exists() or not rerun.exists():
    raise SystemExit('Missing one of the trained_models files')

with open(orig, 'r', encoding='utf-8') as f:
    o = json.load(f)
with open(rerun, 'r', encoding='utf-8') as f:
    r = json.load(f)

oseries = o.get('series_models', {})
rseries = r.get('series_models', {})

changed = []
added = []
for sid, info in rseries.items():
    orig_info = oseries.get(sid)
    if not orig_info:
        added.append(sid)
        continue
    # compare model_spec seasonal component
    orig_ms = orig_info.get('model_spec') or {}
    rerun_ms = info.get('model_spec') or {}
    orig_seasonal = tuple(orig_ms.get('seasonal_order') or ())
    rerun_seasonal = tuple(rerun_ms.get('seasonal_order') or ())
    if orig_seasonal != rerun_seasonal or orig_ms != rerun_ms:
        changed.append({'series_id': sid, 'orig_seasonal': orig_seasonal, 'rerun_seasonal': rerun_seasonal})

# counts
n_orig = len(oseries)
n_rerun = len(rseries)

# fit log summary
fit_entries = []
if fit_log.exists():
    try:
        with open(fit_log, 'r', encoding='utf-8') as f:
            fit_entries = json.load(f)
    except Exception:
        fit_entries = []

failed_entries = []
if failed.exists():
    try:
        with open(failed, 'r', encoding='utf-8') as f:
            failed_entries = json.load(f)
    except Exception:
        failed_entries = []

success_count = sum(1 for e in fit_entries if e.get('success'))
fail_count = sum(1 for e in fit_entries if not e.get('success'))

summary = {
    'n_orig_series': n_orig,
    'n_rerun_series': n_rerun,
    'n_changed': len(changed),
    'n_added': len(added),
    'sample_changed': changed[:20],
    'success_count_in_fit_log': success_count,
    'fit_log_total_entries': len(fit_entries),
    'failed_series_count_file': len(failed_entries),
}

out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print('Summary written to', out)
print('n_orig_series=', n_orig)
print('n_rerun_series=', n_rerun)
print('n_changed=', len(changed))
print('n_added=', len(added))
print('success_count_in_fit_log=', success_count)
print('failed_series_count_file=', len(failed_entries))
