"""Download Cochrane plain-language summary JSON files directly from HF.

Bypasses the `datasets` library's loading-script mechanism (removed in
datasets 3.0) by fetching the raw JSON files over HTTP. Writes the files
into data/raw/cochrane/ for prepare_dataset.py to consume.
"""

import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "cochrane"
BASE_URL = "https://huggingface.co/datasets/GEM/cochrane-simplification/resolve/main"
FILES = ["train.json", "validation.json", "test.json"]


def fetch():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  cached: {dest}")
            continue
        url = f"{BASE_URL}/{name}"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"  saved: {dest} ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    fetch()
    print("Done.")
