import json, hashlib, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
import uuid

BASE = Path(".")
ART = BASE/"artifacts"
LOGS = BASE/"logs"

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)

def read_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()

def log_jsonl(event: dict):
    LOGS.mkdir(exist_ok=True, parents=True)
    event = {"ts": datetime.utcnow().isoformat()+"Z", "run_id": str(uuid.uuid4()), **event}
    with open(LOGS/f"runs_{datetime.utcnow():%Y%m%d}.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False)+"\n")
