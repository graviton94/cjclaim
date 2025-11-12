from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable
import re

def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def month_key_from_year_month(year: int, month: int) -> str:
    return f"{year}-{month:02d}"

def get_latest_file(root: str | Path, pattern: str, recursive: bool = True) -> Optional[Path]:
    root = Path(root)
    if not root.exists():
        return None
    items = root.rglob("*") if recursive else root.glob("*")
    is_regex = any(c in pattern for c in ".^$*+?{}[]|()")
    rx = re.compile(pattern) if is_regex else None
    candidates = []
    for p in items:
        if not p.is_file():
            continue
        name = str(p.relative_to(root)).replace("\\", "/")
        ok = bool(rx.search(name)) if rx else (pattern in name)
        if ok:
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def read_lines(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
