import traceback
from pathlib import Path
import pandas as pd
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

try:
    df = make_good_df()
    g = sv.to_monthly(df)
    print('INDEX TYPE:', type(g.index))
    print('INDEX:', g.index)
    try:
        print('FREQ:', g.index.freq)
        print('FREQSTR:', g.index.freqstr)
    except Exception as e:
        print('Error reading freq:', e)
except Exception:
    traceback.print_exc()
