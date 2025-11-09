# -*- coding: utf-8 -*-
import io
import csv
import pandas as pd
from typing import Dict, Union, IO
from src.constants import SCHEMA_MIN
from pathlib import Path

class SchemaError(Exception):
    """스키마 오류."""
    pass

# ---- 새로 추가: 인코딩/구분자 강건 로더 ----
_CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]

def _read_csv_any_encoding(src: Union[str, bytes, IO[bytes], IO[str]]) -> pd.DataFrame:
    """
    파일 경로(str) 또는 Streamlit UploadedFile(바이너리 스트림)을 받아
    인코딩/구분자를 자동 판별해 DataFrame으로 로드.
    """
    # 1) 바이트 버퍼 확보
    if hasattr(src, "read"):          # file-like (e.g., UploadedFile)
        raw = src.read()
        # Streamlit UploadedFile은 read() 후 포인터가 끝으로 가므로, 이후 재사용시 src.seek(0) 필요
    elif isinstance(src, (bytes, bytearray)):
        raw = bytes(src)
    else:
        # 파일 경로인 경우 먼저 바이너리로 읽어옴
        with open(src, "rb") as f:
            raw = f.read()

    # 2) 인코딩 후보를 순차 시도하여 텍스트로 디코드
    last_err = None
    decoded = None
    used_encoding = None
    for enc in _CANDIDATE_ENCODINGS:
        try:
            decoded = raw.decode(enc)
            used_encoding = enc
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if decoded is None:
        raise UnicodeDecodeError("unknown", b"", 0, 1, f"CSV 인코딩 판별 실패. 시도={_CANDIDATE_ENCODINGS}, last_err={last_err}")

    # 3) 구분자 추정 (쉼표/탭/세미콜론 등)
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(decoded.splitlines()[0])
        sep = dialect.delimiter
    except Exception:
        # 첫 줄을 보고 콤마/탭/세미콜론 순으로 추정
        head = decoded.splitlines()[0]
        if "\t" in head and head.count("\t") >= head.count(","):
            sep = "\t"
        elif ";" in head and head.count(";") > head.count(","):
            sep = ";"
        else:
            sep = ","

    # 4) 판독
    df = pd.read_csv(io.StringIO(decoded), dtype=str, sep=sep)
    # Streamlit 파일 포인터 복구 (재사용 대비)
    if hasattr(src, "seek"):
        try:
            src.seek(0)
        except Exception:
            pass

    # 인코딩/구분자 정보를 필요하면 로그/디버깅용으로 활용 가능
    df.attrs["used_encoding"] = used_encoding
    df.attrs["used_sep"] = sep
    return df

def _parse_date(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    dt = pd.to_datetime(s, errors="coerce")
    na = dt.isna()
    if na.any():
        dt2 = pd.to_datetime(s[na], format="%Y%m%d", errors="coerce")
        dt[na] = dt2
    if dt.isna().any():
        bad = s[dt.isna()].head(3).tolist()
        raise SchemaError(f"날짜 파싱 실패 예시: {bad}")
    return dt

def load_min_csv(path_or_file: Union[str, IO[bytes]]) -> pd.DataFrame:
    """
    최소 스키마 CSV 로드:
    제품범주2, 플랜트, 제조일자, 중분류, count
    - 인코딩 자동 판별(utf-8-sig/utf-8/cp949/euc-kr/latin1)
    - 구분자 자동 추정(, | \\t | ;)
    """
    df = _read_csv_any_encoding(path_or_file)

    missing = [c for c in SCHEMA_MIN if c not in df.columns]
    if missing:
        raise SchemaError(f"누락 컬럼: {missing}")

    # 타입 정규화
    df = df[list(SCHEMA_MIN.keys())].copy()
    for col, typ in SCHEMA_MIN.items():
        if typ == "date":
            df[col] = _parse_date(df[col])
        elif typ == "int":
            # 천단위 콤마/공백 제거
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce"
            ).fillna(0).astype(int)
        elif typ == "str":
            df[col] = df[col].astype(str).str.strip()

    return df

def write_parquet(df, path):
    """DataFrame을 지정 경로에 parquet으로 저장"""
    import pandas as pd
    df.to_parquet(path, index=False)

def log_jsonl(obj, path="artifacts/logs/pipeline_events.jsonl"):
    """dict 객체를 jsonl 파일에 append 저장"""
    import json
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

ART =  Path("artifacts")
