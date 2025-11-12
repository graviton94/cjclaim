"""Simple test harness for src.schema_validator
Runs a few positive and negative checks and prints PASS/FAIL.
"""
import sys
from pathlib import Path
import pandas as pd

# Ensure repo root is on sys.path so `from src import ...` works when running this script
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from src import schema_validator as sv


def make_good_df():
    return pd.DataFrame({
        "발생일자": ["2023-01-05", "2023-01-12", "2023-02-01"],
        "중분류": ["A", "A", "A"],
        "플랜트": ["P1", "P1", "P1"],
        "제품범주2": ["X", "X", "X"],
        "제조일자": ["2022-12-20", "2022-12-20", "2023-01-20"],
        "count": [1, 2, 3]
    })


def test_validate_ok():
    df = make_good_df()
    try:
        sv.validate_schema(df)
        print('TC-validate-ok: PASS')
    except Exception as e:
        print('TC-validate-ok: FAIL', e)
        return 1
    return 0


def test_validate_missing():
    df = make_good_df().drop(columns=['제품범주2'])
    try:
        sv.validate_schema(df)
        print('TC-validate-missing: FAIL (should have raised)')
        return 1
    except Exception:
        print('TC-validate-missing: PASS')
        return 0


def test_to_monthly():
    df = make_good_df()
    try:
        g = sv.to_monthly(df)
        # index freq must be 'MS' and counts sum to 3 for Jan
        if g.index.freqstr != 'MS':
            print('TC-to_monthly: FAIL freq', g.index.freq)
            return 1
        jan = g.loc[pd.to_datetime('2023-01-01')]
        if jan['count'].sum() != 3:
            print('TC-to_monthly: FAIL counts', jan['count'].sum())
            return 1
        print('TC-to_monthly: PASS')
        return 0
    except Exception as e:
        print('TC-to_monthly: FAIL', e)
        return 1


def main():
    exit_code = 0
    exit_code |= test_validate_ok()
    exit_code |= test_validate_missing()
    exit_code |= test_to_monthly()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
