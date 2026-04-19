"""Evaluate the fine-tuned adapter on the validation split.

Reports four metrics:
  BLEU         — n-gram overlap with reference plain-English
  ROUGE-L      — longest common subsequence overlap
  FK grade     — Flesch-Kincaid US grade level of generated text
                 (lower = more readable; target ~8-10 for general audience)
  len ratio    — generated length / source length
                 (simplification should usually shrink text, ratio < 1)
"""

import json
from pathlib import Path

import sacrebleu
import textstat
from rouge_score import rouge_scorer
from tqdm import tqdm

from inference import generate, load

ROOT = Path(__file__).resolve().parent.parent
VAL_PATH = ROOT / "data" / "processed" / "val.jsonl"
OUT_PATH = ROOT / "outputs" / "evaluation_metrics.json"
SAMPLE_PATH = ROOT / "outputs" / "sample_outputs.md"

MAX_EXAMPLES = 200


def load_val(n):
    with open(VAL_PATH) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:n]


def run():
    rows = load_val(MAX_EXAMPLES)
    model, tokenizer = load()

    predictions, references, sources = [], [], []
    for row in tqdm(rows, desc="generating"):
        predictions.append(generate(model, tokenizer, row["input"]))
        references.append(row["output"])
        sources.append(row["input"])

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(
        scorer.score(r, p)["rougeL"].fmeasure
        for r, p in zip(references, predictions)
    ) / len(predictions)

    fk_grade = sum(textstat.flesch_kincaid_grade(p) for p in predictions) / len(predictions)
    fk_source = sum(textstat.flesch_kincaid_grade(s) for s in sources) / len(sources)

    length_ratio = sum(len(p) / max(len(s), 1) for p, s in zip(predictions, sources)) / len(predictions)

    metrics = {
        "n_examples": len(predictions),
        "bleu": round(bleu, 2),
        "rouge_l": round(rouge_l, 4),
        "fk_grade_generated": round(fk_grade, 2),
        "fk_grade_source": round(fk_source, 2),
        "length_ratio_vs_source": round(length_ratio, 3),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    write_samples(sources, predictions, references)


def write_samples(sources, predictions, references, n=10):
    lines = ["## Sample Outputs", ""]
    for i, (src, pred, ref) in enumerate(zip(sources[:n], predictions[:n], references[:n]), 1):
        lines += [
            f"### Example {i}",
            "",
            "**Clinical source:**  ",
            src,
            "",
            "**Model output:**  ",
            pred,
            "",
            "**Reference plain-English:**  ",
            ref,
            "",
            "---",
            "",
        ]
    SAMPLE_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    run()
