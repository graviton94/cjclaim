# -*- coding: utf-8 -*-
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(r"C:\cjclaim")
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"            # (후속) DuckDB 사용 예정
WEEK_FREQ = "W-MON"                 # 월요일 시작 주차
SEASONAL_PERIOD = 52
EWS_WINDOW_RECENT = 4
EWS_WINDOW_BASE = 12

SCHEMA_MIN = {
    "제품범주2": "str",
    "플랜트": "str",
    "제조일자": "date",
    "중분류(보정)": "str",
    "count": "int",
}

@dataclass
class SplitConfig:
    train_start: str = "2021-01-01"
    train_end:   str = "2024-12-31"
    test_start:  str = "2025-01-01"
    test_end:    str = "2025-12-31"
