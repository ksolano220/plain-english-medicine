## Plain English Medicine — V1 LoRA iteration (lessons learned)

A first-pass LoRA fine-tune of **Qwen2.5-1.5B-Instruct** on the [Cochrane Plain Language Summaries](https://huggingface.co/datasets/GEM/cochrane-simplification) dataset, intended to rewrite clinical and biomedical text in plain English.

**This project is kept public as a documented learning artifact.** The successor, which applies the lessons from this run, is here: **[symptom-triage-coach](https://github.com/ksolano220/symptom-triage-coach)** (the live deployed demo).

### What went wrong

The Cochrane PLS dataset's pairs are **(research abstract → Cochrane plain-language review summary)**. A Cochrane PLS is written from a full systematic review, not from the abstract alone, so the targets contain aggregated facts the inputs don't support: trial counts, bias assessments, risk ratios, confidence intervals.

Fine-tuning on those pairs taught the model to generate review-formatted output including **fabricated trial statistics** — the classic task/data mismatch.

Example of the failure mode — given a single clinical sentence as input:

> *Patient presents with acute exacerbation of chronic obstructive pulmonary disease.*

The trained adapter confidently produced:

> *We identified 10 trials involving 2,345 patients... Two trials were at low risk of bias and eight at high risk of bias. In one trial comparing oral corticosteroids to placebo, there was no difference between groups in overall mortality (risk ratio (RR) 0.88, 95% CI 0.50 to 1.56)...*

None of those trials, patient counts, or statistics exist. The model learned a genre (systematic-review PLS) and hallucinated content to fit it.

### What I took from it

1. **A dataset whose targets depend on inputs the model will never see at inference time will train the model to hallucinate those dependencies.** The failure wasn't model quality — it was data/task mismatch.
2. **Medical AI outputs with fabricated statistics are a real liability even with disclaimers.** I decided a public demo wasn't responsible to ship, and took down the Hugging Face Space.
3. **The successor project ([symptom-triage-coach](https://github.com/ksolano220/symptom-triage-coach)) uses synthetic training data generated to a strict JSON schema**, with every pair validated before training. Hallucinated content literally cannot fit the output shape.

### What's still useful in this repo

The end-to-end pipeline code is solid and was reused in V2:
- Colab notebook with HF Hub push
- LoRA training loop (PEFT + TRL SFTTrainer, 4-bit QLoRA on T4)
- Gradio Space scaffolding
- Evaluation script (BLEU, ROUGE, Flesch-Kincaid)

See `src/` and `notebooks/train_colab.ipynb`.

### Artifacts

- Trained adapter still on HF: [huggingface.co/ksolano220/plain-english-medicine](https://huggingface.co/ksolano220/plain-english-medicine)
- No live Gradio demo (retired; see [symptom-triage-coach](https://huggingface.co/spaces/ksolano220/symptom-triage-coach) for the working demo)

### Tools Used

- PyTorch, Transformers, PEFT, TRL (SFTTrainer)
- bitsandbytes (4-bit quantization)
- Hugging Face Datasets, Hub
- Gradio (Space scaffolding)
- sacreBLEU, rouge-score, textstat (evaluation)
