from pathlib import Path
import pandas as pd, re

ROOT = Path("artifacts")
PATTERNS = [
    (ROOT/"metrics", re.compile(r"^monthly_ts_\d{6}\.parquet$")),
    (ROOT/"forecasts", re.compile(r"^forecast_\d{4}_\d{2}\.parquet$")),
    (ROOT/"metrics", re.compile(r"^reconcile_\d{4}_\d{2}\.parquet$")),
]
def is_valid_parquet(p: Path) -> bool:
    try:
        _ = pd.read_parquet(p).head(1)
        return True
    except Exception:
        return False

def run():
    bad = []
    for base, rx in PATTERNS:
        if not base.exists(): continue
        for p in base.iterdir():
            if rx.match(p.name):
                if not is_valid_parquet(p):
                    bad.append(p)
    for p in bad:
        print(f"[DEL] broken parquet: {p}")
        try: p.unlink()
        except Exception as e: print(f"  -> delete failed: {e}")
    print(f"Checked. Removed {len(bad)} broken files.")

if __name__ == "__main__":
    run()
