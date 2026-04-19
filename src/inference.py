"""Load base + LoRA adapter and generate plain-English versions of clinical text."""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER = "ksolano220/plain-english-medicine"

SYSTEM_PROMPT = (
    "You are a medical communication assistant. Rewrite the clinical or "
    "biomedical text the user provides in plain English a non-expert adult "
    "could understand. Preserve all medical facts. Do not add advice or "
    "disclaimers. Keep it concise."
)


def load(adapter_id: str = DEFAULT_ADAPTER, quantize: bool = True):
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model = PeftModel.from_pretrained(base, adapter_id)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, clinical_text: str, max_new_tokens: int = 256) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": clinical_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return generated.strip()


if __name__ == "__main__":
    model, tokenizer = load()
    example = (
        "Patient presents with acute exacerbation of chronic obstructive "
        "pulmonary disease requiring supplemental oxygen and inhaled bronchodilators."
    )
    print("INPUT:", example)
    print("OUTPUT:", generate(model, tokenizer, example))
