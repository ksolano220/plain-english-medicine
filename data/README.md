## Data

Training data for the plain-English medicine LoRA comes from the Cochrane Plain Language Summaries, a public biomedical text-simplification dataset freely licensed for research use.

### Source

| Dataset | Size | What it is |
|---------|------|------------|
| [Cochrane Plain Language Summaries](https://huggingface.co/datasets/GEM/cochrane-simplification) | ~4,500 pairs | Cochrane systematic review abstracts paired with the reviewer-written plain-language summary intended for the general public |

### Folder layout

```
data/
├── README.md             # this file
├── raw/                  # downloaded HF datasets (gitignored)
├── processed/            # unified JSONL of {input, output} pairs (gitignored)
└── sample.jsonl          # 20 example pairs checked into the repo for reference
```

### How pairs are formatted

Every example is a single JSONL row with two fields:

```json
{"input": "Clinical/biomedical text with jargon.", "output": "The same meaning rewritten for a general audience."}
```

The training script wraps these in an instruction template before tokenization (see `src/prepare_dataset.py`).

### Reproducing the dataset

```bash
python src/fetch_data.py        # downloads raw HF datasets into data/raw/
python src/prepare_dataset.py   # unifies into data/processed/train.jsonl + val.jsonl
```

### Ethical note

This dataset contains only public biomedical research text, not real patient data. The trained model is intended for educational and research use and is not a substitute for medical advice from a licensed clinician.
