from __future__ import annotations
from pathlib import Path
import pandas as pd, os, tempfile

def _write_parquet_atomic(df: pd.DataFrame, out_path: str):
    p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=p.parent, delete=False, suffix=".parquet") as tmp:
        tmpn = tmp.name
    try:
        df.to_parquet(tmpn, index=False)
        _ = pd.read_parquet(tmpn).head(1)  # sanity check
        os.replace(tmpn, p)
    except Exception:
        try: os.remove(tmpn)
        except: pass
        raise

def build_monthly_timeseries(curated_parquet: str, out_parquet: str) -> dict:
    # is_flat_series: 단일값/패딩 탐지 (bool 보장)
    def is_flat(s):
        v = s.dropna().unique()
        return len(v) <= 1

    try:
        df = pd.read_parquet(curated_parquet)
    except Exception:
        df = pd.DataFrame()

    # 표준 컬럼명 정의
    required_cols = ["series_id", "yyyymm", "year", "month", "claim_count"]
    # 컬럼명 표준화: 한글→영문, count→claim_count, ym→yyyymm
    rename_map = {
        "발생건수": "claim_count",
        "플랜트": "plant",
        "제품범주2": "product_category2",
        "중분류": "mid_category",
        "제조일자": "manufacture_date",
        "발생일자": "event_date"
    }
    if not df.empty:
        df.rename(columns=rename_map, inplace=True)
        # 날짜 컬럼 변환
        for c in ["event_date","manufacture_date"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        # series_id 생성: 표준 컬럼명 사용
        # If a canonical 'series_id' already exists in the curated data, keep it.
        id_cols = ["plant","product_category2","mid_category"]
        if "series_id" in df.columns:
            # ensure series_id is string
            df["series_id"] = df["series_id"].astype(str)
        else:
            available_id_cols = [c for c in id_cols if c in df.columns]
            missing_id_cols = [c for c in id_cols if c not in df.columns]
            if len(available_id_cols) > 0:
                # Fill missing parts with 'unknown' to avoid dropping rows
                for c in available_id_cols:
                    df[c] = df[c].fillna('unknown').astype(str)
                # For any missing id columns that don't exist in the DataFrame, create them as 'unknown'
                for c in missing_id_cols:
                    df[c] = 'unknown'
                # Now build series_id from the three components to keep consistency
                df["series_id"] = df[["plant","product_category2","mid_category"]].astype(str).agg("_".join, axis=1)
                if missing_id_cols:
                    print(f"[경고] series_id 생성에 일부 컬럼 누락: {missing_id_cols}. 기본값 'unknown'으로 채워집니다.")
            else:
                # No id columns at all — use fallback 'unknown' for all rows
                df["series_id"] = 'unknown'
        # 월 정보 생성: yyyymm (정수)
        if "manufacture_date" in df.columns:
            df["yyyymm"] = (df["manufacture_date"].dt.year*100 + df["manufacture_date"].dt.month).astype(int)
        elif {"year","month"} <= set(df.columns):
            df["yyyymm"] = (df["year"].astype(int)*100 + df["month"].astype(int)).astype(int)
        else:
            raise KeyError("월 키 생성에 필요한 'manufacture_date' 또는 'year'/'month' 컬럼이 누락됨")
        # claim_count 컬럼 존재 확인 및 결측치 처리
        if "claim_count" not in df.columns:
            if "count" in df.columns:
                df["claim_count"] = df["count"]
            else:
                df["claim_count"] = 1
        df["claim_count"] = df["claim_count"].fillna(0).astype(int)
        # 집계: series_id, yyyymm별로 claim_count 합산
        monthly = (
            df.groupby(["series_id","yyyymm"], as_index=False)["claim_count"].sum()
              .rename(columns={"claim_count":"y"})
        )
        # 정수형 분리
        monthly["yyyymm"] = monthly["yyyymm"].astype(str).str.replace(r"[^0-9]","",regex=True).astype(int)
        # Ensure year/month columns exist (derive from yyyymm)
        try:
            monthly["year"]  = (monthly["yyyymm"] // 100).astype(int)
            monthly["month"] = (monthly["yyyymm"] % 100).astype(int)
        except Exception:
            # if derivation fails, ensure columns exist with safe defaults
            if 'year' not in monthly.columns:
                monthly['year'] = pd.Series([None] * len(monthly))
            if 'month' not in monthly.columns:
                monthly['month'] = pd.Series([None] * len(monthly))
        # 필수 컬럼 보장: 누락시 None 또는 적절한 기본값으로 채움
        for col in ["series_id","yyyymm","year","month","y"]:
            if col not in monthly.columns:
                monthly[col] = None
        # yyyymm 기준 정렬
        monthly["is_flat_series"] = monthly.groupby("series_id")["y"].transform(is_flat).astype(bool)
    else:
        # 입력이 비어있으면 표준 스키마로 빈 DataFrame 생성
        monthly = pd.DataFrame(columns=["series_id","yyyymm","year","month","y","is_flat_series"])
    # dtype 고정 (빈 DF도 동일 스키마)
    required = {
        "series_id": "string",
        "yyyymm": "int64",
        "year": "int32",
        "month": "int32",
        "y": "float64",
        "is_flat_series": "bool",
    }
    monthly = monthly.astype(required)
    for c, dt in required.items():
        if c not in monthly.columns:
            monthly[c] = pd.Series(dtype=dt)
        else:
            try:
                monthly[c] = monthly[c].astype(dt)
            except Exception:
                pass
    monthly = monthly[list(required.keys())]
    _write_parquet_atomic(monthly, out_parquet)
    return {
        "train_monthly": out_parquet,
        "rows": int(len(monthly)),
        "flat_count": int(monthly["is_flat_series"].sum()) if "is_flat_series" in monthly.columns else 0,
        "input_rows": int(df.shape[0]) if not df.empty else 0,
        "used_rows": int(monthly.shape[0]),
        "excluded_rows": int(df.shape[0] - monthly.shape[0]) if not df.empty else 0
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python build_monthly_timeseries.py <curated_parquet> <out_parquet>")
        sys.exit(1)
    build_monthly_timeseries(sys.argv[1], sys.argv[2])
