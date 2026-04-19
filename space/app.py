"""Hugging Face Space entry point for Plain English Medicine.

Self-contained Gradio app. Loads the Qwen2.5-1.5B-Instruct base model
and the LoRA adapter from the Hub, runs inference on free-tier CPU.
No src/ package dependency so the Space can deploy from a flat layout.
"""

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_ID = "ksolano220/plain-english-medicine"

SYSTEM_PROMPT = (
    "You are a medical communication assistant. Rewrite the clinical or "
    "biomedical text the user provides in plain English a non-expert adult "
    "could understand. Preserve all medical facts. Do not add advice or "
    "disclaimers. Keep it concise."
)

EXAMPLES = [
    "Patient presents with acute exacerbation of chronic obstructive pulmonary disease requiring supplemental oxygen and inhaled bronchodilators.",
    "ECG demonstrates ST-elevation in leads II, III, and aVF consistent with inferior wall myocardial infarction.",
    "Labs are notable for creatinine of 2.8 and potassium of 5.9, consistent with acute kidney injury and hyperkalemia.",
    "MRI brain reveals a 2.3 cm enhancing lesion in the left frontal lobe with surrounding vasogenic edema, concerning for primary neoplasm.",
    "The biopsy is negative for malignancy but demonstrates chronic inflammation and mild dysplasia.",
]

DESCRIPTION = """
# Plain English Medicine

Rewrites clinical and biomedical text in plain language a non-expert can understand. Built on Qwen2.5-1.5B fine-tuned with LoRA on the Cochrane Plain Language Summaries dataset.

**Not medical advice.** For information and research use only.

First request takes ~60 seconds while the model loads. Subsequent requests are ~10–20 seconds on free-tier CPU.
"""

print("Loading base model and adapter (this takes ~60s)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
model = PeftModel.from_pretrained(base, ADAPTER_ID)
model.eval()
print("Model ready.")


def simplify(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return generated.strip()


with gr.Blocks(title="Plain English Medicine") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            source = gr.Textbox(
                label="Clinical text",
                lines=6,
                placeholder="Paste clinical notes, radiology reports, or a biomedical abstract...",
            )
            run_btn = gr.Button("Translate to plain English", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Plain English", lines=6, interactive=False)

    gr.Examples(examples=EXAMPLES, inputs=source)
    run_btn.click(simplify, inputs=source, outputs=output)


if __name__ == "__main__":
    demo.launch()
