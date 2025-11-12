from __future__ import annotations
from pathlib import Path
import json, statistics as st
from typing import Iterable, Dict, Any

def _iter_series_stats(json_paths: Iterable[Path], sample: int|None = None):
    paths = list(json_paths)
    if sample is not None:
        paths = paths[:sample]
    for p in paths:
        try:
            obj = json.loads(Path(p).read_text(encoding="utf-8"))
            # Robust yearmonth extraction
            ym = None
            if "yearmonth" in obj:
                ym = obj["yearmonth"]
            elif "data" in obj and isinstance(obj["data"], list):
                # If yearmonth missing, build from year/month
                ym = []
                for d in obj["data"]:
                    if "yearmonth" in d:
                        ym.append(int(str(d["yearmonth"]).replace("-", "").replace("/", "")))
                    elif "year" in d and "month" in d:
                        yyyymm = int(d["year"])*100 + int(d["month"])
                        ym.append(yyyymm)
                if not ym:
                    continue
            elif "data" in obj and isinstance(obj["data"], dict):
                ym = [int(str(k).replace("-", "").replace("/", "")) for k in obj["data"].keys()]
            if ym is None or not ym:
                continue
            y = obj.get("values")
            if y is None and "data" in obj and isinstance(obj["data"], list):
                y = [d.get("claim_count", 0) for d in obj["data"]]
            n = len(y) if y else 0
            nonzero = sum(1 for v in y if v != 0) if y else 0
            avg = (sum(float(v) for v in y)/n) if n else 0.0
            yield {
                "min_ym": min(ym), "max_ym": max(ym),
                "avg": avg, "nonzero_ratio": (nonzero/n if n else 0.0),
                "n": n
            }
        except Exception:
            continue

def scan_series_distribution(json_paths: Iterable[Path], sample: int|None = None) -> Dict[str, Any]:
    stats = list(_iter_series_stats(json_paths, sample))
    if not stats:
        return {"p40_avg": 0.2, "p40_nonzero": 0.15, "min_ym": 201001, "max_ym": 201001}
    avgs = sorted(s["avg"] for s in stats)
    nzrs = sorted(s["nonzero_ratio"] for s in stats)
    def p40(a): 
        k = max(0, min(len(a)-1, int(0.4*(len(a)-1))))
        return a[k] if a else 0.0
    return {
        "p40_avg": p40(avgs),
        "p40_nonzero": p40(nzrs),
        "min_ym": min(s["min_ym"] for s in stats),
        "max_ym": max(s["max_ym"] for s in stats)
    }

def infer_train_window(min_ym: int, max_ym: int, lookback_months: int):
    # ym: yyyymm 정수. lookback_months만큼 뒤로 이동
    y = max_ym // 100; m = max_ym % 100
    lm = lookback_months - 1
    y_start = y - (lm // 12)
    m_shift = m - (lm % 12)
    while m_shift <= 0:
        y_start -= 1; m_shift += 12
    start_ym = y_start * 100 + m_shift
    return max(min_ym, start_ym), max_ym
