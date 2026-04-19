"""Unify raw HF datasets into a single JSONL of {input, output} pairs.

Reads from data/raw/ (populated by fetch_data.py) and writes:
  data/processed/train.jsonl
  data/processed/val.jsonl

Each line is one JSON object with:
  input:  clinical/biomedical source text
  output: plain-English version
"""

import json
import random
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

VAL_FRACTION = 0.1
SEED = 42


def from_cochrane():
    ds = load_from_disk(str(RAW_DIR / "cochrane"))
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        for row in ds[split]:
            src = (row.get("source") or "").strip()
            tgt = (row.get("target") or "").strip()
            if src and tgt:
                yield {"input": src, "output": tgt}


def collect():
    return list(from_cochrane())


def split(pairs):
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
    train, val = split(pairs)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)
    print(f"Wrote {len(train)} train / {len(val)} val to {OUT_DIR}")
