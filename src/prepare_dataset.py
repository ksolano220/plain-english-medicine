"""Unify the Cochrane JSON files into a single JSONL of {input, output} pairs.

Reads from data/raw/cochrane/ (populated by fetch_data.py) and writes:
  data/processed/train.jsonl
  data/processed/val.jsonl

Each line is one JSON object with:
  input:  clinical/biomedical source text
  output: plain-English version
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "cochrane"
OUT_DIR = ROOT / "data" / "processed"

VAL_FRACTION = 0.1
SEED = 42


def read_records(path: Path):
    """Parse a JSON file that is either a JSON array or JSONL. Robust to both."""
    content = path.read_text().strip()
    if not content:
        return []
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def from_cochrane():
    for name in ("train.json", "validation.json", "test.json"):
        path = RAW_DIR / name
        if not path.exists():
            print(f"  missing: {path}")
            continue
        records = read_records(path)
        print(f"  {name}: {len(records)} records")
        for row in records:
            src = (row.get("source") or "").strip()
            tgt = (row.get("target") or "").strip()
            if src and tgt:
                yield {"input": src, "output": tgt}


def collect():
    return list(from_cochrane())


def split_pairs(pairs):
    random.Random(SEED).shuffle(pairs)
    cut = int(len(pairs) * (1 - VAL_FRACTION))
    return pairs[:cut], pairs[cut:]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    pairs = collect()
    print(f"Collected {len(pairs)} pairs.")
    if not pairs:
        raise SystemExit("No pairs collected. Run fetch_data.py first.")
    train, val = split_pairs(pairs)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)
    print(f"Wrote {len(train)} train / {len(val)} val to {OUT_DIR}")
