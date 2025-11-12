import pandas as pd
import json
from pathlib import Path

# 입력/출력 경로
PARQUET_PATH = "artifacts/metrics/claims_monthly_2021_2023.parquet"
FEATURE_DIR = Path("data/features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_parquet(PARQUET_PATH)
    # series_id 기준 그룹화
    for series_id, group in df.groupby("series_id"):
        # 안전한 파일명 생성
        safe_filename = (
            series_id.replace('/', '_').replace('\\', '_').replace(':', '_')
            .replace('|', '_').replace('?', '_').replace('*', '_')
            .replace('<', '_').replace('>', '_').replace('"', '_')
        )
        json_path = FEATURE_DIR / f"{safe_filename}.json"
        # 월별 데이터 정리
        data = []
        for _, row in group.iterrows():
            data.append({
                "year": int(row["year"]),
                "month": int(row["month"]),
                "claim_count": float(row["claim_count"]),
            })
        # 정렬
        data = sorted(data, key=lambda x: (x["year"], x["month"]))
        # 저장 (series_id 포함)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"series_id": series_id, "data": data}, f, ensure_ascii=False, indent=2)
        print(f"[UPDATED] {json_path} ({len(data)} records)")

if __name__ == "__main__":
    main()
