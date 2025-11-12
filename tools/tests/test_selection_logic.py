"""Simple test script for selection_utils.prefer_seasonal_by_aic.

This is not a full pytest test but a minimal script you can run with python to
quickly verify logic in environments without pytest installed.
"""
import sys
import importlib.util
from pathlib import Path

# Load selection_utils directly from src/ to avoid package import issues in test env
spec = importlib.util.spec_from_file_location(
    'selection_utils', str(Path(__file__).resolve().parents[2] / 'src' / 'selection_utils.py')
)
selection_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection_utils)
prefer_seasonal_by_aic = selection_utils.prefer_seasonal_by_aic


def case_promote():
    chosen = {'spec': {'seasonal_order': (0, 0, 0, 0)}, 'aic': 100.0}
    candidates = [
        {'spec': {'seasonal_order': (0, 0, 0, 0)}, 'aic': 100.0},
        {'spec': {'seasonal_order': (1, 1, 1, 12)}, 'aic': 101.5},
        {'spec': {'seasonal_order': (1, 1, 1, 12)}, 'aic': 101.9},
    ]
    # seasonal_best has aic=101.5 which is <= 100 + 2 -> should promote
    out = prefer_seasonal_by_aic(chosen, candidates, delta=2.0)
    assert out.get('spec', {}).get('seasonal_order') != (0, 0, 0, 0), 'Expected promotion to seasonal'


def case_no_promote():
    chosen = {'spec': {'seasonal_order': (0, 0, 0, 0)}, 'aic': 90.0}
    candidates = [
        {'spec': {'seasonal_order': (1, 1, 1, 12)}, 'aic': 100.5},
    ]
    out = prefer_seasonal_by_aic(chosen, candidates, delta=2.0)
    # seasonal aic 100.5 > 90 + 2 -> no promotion
    assert out.get('spec', {}).get('seasonal_order') == (0, 0, 0, 0), 'Expected no promotion'


def main():
    case_promote()
    case_no_promote()
    print('selection logic tests passed')


if __name__ == '__main__':
    main()
