"""Create curated monthly parquet and features parquet from a raw CSV.

Usage (PowerShell):
  cd c:\cjclaim\quality-cycles
  python tools\create_parquets_from_csv.py path\to\raw.csv

What it does:
- Attempts to map common column names to required schema
- Validates schema using src.schema_validator.validate_schema
- Aggregates to monthly series (fills missing months with zeros per series)
- Writes data/curated/claims_monthly.parquet with columns: series_id, year, month, yyyymm, claim_count
- Writes data/features/cycle_features.parquet containing simple series-level features

If your CSV columns use different names, pass a mapping JSON file as second arg or edit the mapping dict below.
"""
import sys
from pathlib import Path
import json
import pandas as pd

# local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import schema_validator as sv

# default mapping: try to detect common English/Korean names
DEFAULT_MAP = {
    'date': '발생일자',
    'occur_date': '발생일자',
    '발생일자': '발생일자',
    'manufacture_date': '제조일자',
    '제조일자': '제조일자',
    'category': '중분류',
    '중분류': '중분류',
    'plant': '플랜트',
    '플랜트': '플랜트',
    'product_cat2': '제품범주2',
    '제품범주2': '제품범주2',
    'count': 'count',
    'claims': 'count',
    'claim_count': 'count',
}


def infer_and_rename(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    # Lowercase keys for matching convenience
    cols = list(df.columns)
    rename = {}
    for c in cols:
        cl = c.strip()
        key = cl
        # try exact matches
        if key in mapping:
            rename[c] = mapping[key]
            continue
        # try lowercased matches
        kl = key.lower()
        if kl in mapping:
            rename[c] = mapping[kl]
            continue
        # heuristics
        if 'date' in kl and '발생' in '':
            if '발생' not in mapping.values():
                rename[c] = '발생일자'
        if 'manufact' in kl or '제조' in kl:
            rename[c] = '제조일자'
        if 'count' in kl or 'claim' in kl:
            rename[c] = 'count'
    if rename:
        df = df.rename(columns=rename)
    return df


def main(argv):
    if len(argv) < 2:
        print('Usage: python tools\\create_parquets_from_csv.py <raw_csv_path> [colmap.json]')
        return 2
    raw_csv = Path(argv[1])
    if not raw_csv.exists():
        print('Raw CSV not found:', raw_csv)
        return 3

    colmap = DEFAULT_MAP.copy()
    if len(argv) > 2:
        mpath = Path(argv[2])
        if mpath.exists():
            with open(mpath, 'r', encoding='utf-8') as f:
                user_map = json.load(f)
            colmap.update(user_map)

    df = pd.read_csv(raw_csv)
    print('Loaded', len(df), 'rows from', raw_csv)

    df = infer_and_rename(df, colmap)

    # Quick column check
    print('Columns after rename:', list(df.columns)[:20])

    # Validate schema (will raise a ValueError if missing/extra)
    try:
        sv.validate_schema(df)
    except Exception as e:
        print('Schema validation failed:', e)
        print('You may need to provide a column mapping JSON to map your CSV columns to the required schema.')
        return 4

    # to_monthly returns a DataFrame indexed by 발생월 with columns including group keys and count
    monthly = sv.to_monthly(df)
    # monthly index is DatetimeIndex at month-start; reset and create flat curated frame
    m = monthly.reset_index()
    # ensure columns: 발생월, 중분류, 플랜트, 제품범주2, 제조일자, count
    m = m.rename(columns={'발생월': '발생월', 'count': 'claim_count'})

    # create series_id as combo
    m['series_id'] = m[['플랜트','제품범주2','중분류']].astype(str).agg('_'.join, axis=1)
    m['year'] = m['발생월'].dt.year
    m['month'] = m['발생월'].dt.month
    m['yyyymm'] = (m['year']*100 + m['month']).astype(int)

    curated_dir = Path('data/curated')
    curated_dir.mkdir(parents=True, exist_ok=True)
    curated_path = curated_dir / 'claims_monthly.parquet'

    # Write curated parquet with columns needed by pipeline_train
    curated_df = m[['series_id','year','month','yyyymm','claim_count']]
    curated_df.to_parquet(curated_path, index=False)
    print('Wrote curated monthly parquet:', curated_path)

    # Build lightweight features per series
    feats = (
        curated_df
        .groupby('series_id')['claim_count']
        .agg(n_months='count',
             total='sum',
             nonzero_months=lambda x: (x>0).sum(),
             hist_mean=lambda x: x.mean(),
             hist_std=lambda x: x.std(),
             last_value=lambda x: x.iloc[-1])
        .reset_index()
    )
    feats['nonzero_pct'] = feats['nonzero_months'] / feats['n_months'] * 100

    features_dir = Path('data/features')
    features_dir.mkdir(parents=True, exist_ok=True)
    features_path = features_dir / 'cycle_features.parquet'
    feats.to_parquet(features_path, index=False)
    print('Wrote features parquet:', features_path)

    print('Done.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
