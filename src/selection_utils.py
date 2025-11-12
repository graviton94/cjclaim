"""Selection utilities for model candidate post-processing.

Provides a small helper to prefer seasonal candidates when their AIC is within
a delta of the chosen (loss-minimizing) candidate.

This is intentionally tiny and easily testable.
"""
from typing import Dict, List, Any


def prefer_seasonal_by_aic(chosen: Dict[str, Any], successful_candidates: List[Dict[str, Any]], delta: float = 2.0):
    """Return a candidate to use, preferring a seasonal candidate when its AIC
    is within `delta` of the chosen candidate's AIC.

    - chosen: the candidate dict selected by primary metric (e.g., loss).
    - successful_candidates: list of candidate dicts (each should have 'spec' and optionally 'aic').
    - delta: numeric threshold for AIC difference.

    If chosen is non-seasonal and there exists a seasonal candidate with aic <= chosen_aic + delta,
    the seasonal candidate with lowest aic is returned. Otherwise the original chosen is returned.
    """
    if not chosen:
        return chosen

    try:
        best_aic = chosen.get('aic', None)
        # seasonal candidates: those with a non-zero seasonal_order
        seasonal_candidates = [c for c in successful_candidates if c.get('spec', {}).get('seasonal_order') and c.get('spec', {}).get('seasonal_order') != (0, 0, 0, 0)]
        # if chosen is non-seasonal and we have seasonal candidates, consider promotion
        if chosen.get('spec', {}).get('seasonal_order') == (0, 0, 0, 0) and seasonal_candidates:
            seasonal_with_aic = [c for c in seasonal_candidates if c.get('aic') is not None]
            if seasonal_with_aic:
                seasonal_best = min(seasonal_with_aic, key=lambda x: x['aic'])
                if best_aic is None or (seasonal_best['aic'] <= best_aic + float(delta)):
                    return seasonal_best
    except Exception:
        # on any unexpected error, return the original chosen candidate
        return chosen

    return chosen
