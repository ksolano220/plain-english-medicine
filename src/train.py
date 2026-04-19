"""LoRA fine-tuning for plain-English medical text generation.

Trains LoRA adapters on top of a 4-bit quantized Qwen2.5-1.5B-Instruct
base model, targeting attention and MLP projection matrices. Designed
to run on a single Colab T4 GPU (16 GB) in 30 to 60 minutes.

The base model weights stay frozen. Only the LoRA adapters update,
producing a ~20 MB artifact that can be pushed to Hugging Face Hub.
The small base model also runs on free-tier CPU for inference, so the
live Gradio demo on Hugging Face Spaces stays on free infrastructure.
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "outputs/lora_weights"

SYSTEM_PROMPT = (
    "You are a medical communication assistant. Rewrite the clinical or "
    "biomedical text the user provides in plain English a non-expert adult "
    "could understand. Preserve all medical facts. Do not add advice or "
    "disclaimers. Keep it concise."
)


def format_example(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"messages": messages}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(train_path, val_path):
    train_rows = [format_example(r) for r in load_jsonl(train_path)]
    val_rows = [format_example(r) for r in load_jsonl(val_path)]
    return Dataset.from_list(train_rows), Dataset.from_list(val_rows)


def load_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def attach_lora(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(train_path, val_path, hub_repo_id=None):
    train_ds, val_ds = build_dataset(train_path, val_path)
    model, tokenizer = load_base_model()
    model = attach_lora(model)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        report_to="none",
        push_to_hub=bool(hub_repo_id),
        hub_model_id=hub_repo_id,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        max_seq_length=1024,
        packing=False,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    if hub_repo_id:
        trainer.push_to_hub()
    return trainer


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    train(
        train_path=root / "data" / "processed" / "train.jsonl",
        val_path=root / "data" / "processed" / "val.jsonl",
    )
