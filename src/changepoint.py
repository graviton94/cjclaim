# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
try:
    import ruptures as rpt
except Exception:
    rpt = None

def detect_changepoints(y: pd.Series, max_cp: int = 5) -> pd.DataFrame:
    if rpt is None or len(y) < 10:
        return pd.DataFrame(columns=["idx"])
    x = y.values.astype(float).reshape(-1, 1)
    model = rpt.Pelt(model="rbf").fit(x)
    pen = np.log(len(y)) * (np.std(x) + 1e-6)
    bkps = model.predict(pen=pen)
    idxs = [b-1 for b in bkps[:-1] if 0 < b-1 < len(y)-1][:max_cp]
    return pd.DataFrame({"idx": idxs})
