"""Download raw biomedical text simplification datasets from Hugging Face.

Writes Arrow/Parquet files into data/raw/. The prepare_dataset.py step
unifies these into a single JSONL of {input, output} pairs for training.
"""

from pathlib import Path

from datasets import load_dataset

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_cochrane():
    print("Downloading Cochrane plain-language summaries...")
    ds = load_dataset("GEM/cochrane_simplification")
    out = RAW_DIR / "cochrane"
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"  saved to {out}")


def fetch_plaba():
    print("Downloading PLABA biomedical abstracts...")
    ds = load_dataset("bigbio/plaba", "plaba_source")
    out = RAW_DIR / "plaba"
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"  saved to {out}")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetch_cochrane()
    fetch_plaba()
    print("Done.")
