"""Gradio demo for the plain-English medicine LoRA.

Deployed as a Hugging Face Space:
https://huggingface.co/spaces/ksolano220/plain-english-medicine
"""

import gradio as gr

from src.inference import generate, load

ADAPTER_ID = "ksolano220/plain-english-medicine"

EXAMPLES = [
    "Patient presents with acute exacerbation of chronic obstructive pulmonary disease requiring supplemental oxygen and inhaled bronchodilators.",
    "ECG demonstrates ST-elevation in leads II, III, and aVF consistent with inferior wall myocardial infarction. Patient taken emergently to the catheterization lab.",
    "Labs are notable for creatinine of 2.8 and potassium of 5.9, consistent with acute kidney injury and hyperkalemia.",
    "MRI brain reveals a 2.3 cm enhancing lesion in the left frontal lobe with surrounding vasogenic edema, concerning for primary neoplasm.",
    "The biopsy is negative for malignancy but demonstrates chronic inflammation and mild dysplasia.",
]

DESCRIPTION = """
# Plain English Medicine

This tool rewrites clinical and biomedical text in plain language that a non-expert can understand. It uses a Llama 3.1 8B model fine-tuned with LoRA on the Cochrane Plain Language Summaries and PLABA datasets.

**Not medical advice.** For information only.
"""

print("Loading model (this takes ~60s on first launch)...")
model, tokenizer = load(ADAPTER_ID)
print("Model ready.")


def simplify(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return generate(model, tokenizer, text)


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
