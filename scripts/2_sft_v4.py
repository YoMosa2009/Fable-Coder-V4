import torch
import os
import sys
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

print("=== Phase 2: Starting Fable-Coder V4 SFT Training (Replay-Anchored LoRA) ===", flush=True)

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
TRAIN_DATA = "/content/fable_v4_sft_train.jsonl"
VAL_DATA = "/content/fable_v4_sft_val.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/Fable_Coder_V4/checkpoints_sft"
FINAL_LORA_DIR = "/content/drive/MyDrive/Fable_Coder_V4/fable_v4_lora_sft"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_LORA_DIR, exist_ok=True)

print("1. Loading Tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("2. Configuring NF4 Quantization & Loading Base Model...", flush=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

print("3. Attaching LoRA to ALL 7 Projections (Rank=32, Alpha=64)...", flush=True)
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("4. Loading and formatting 4-Pillar dataset...", flush=True)
def format_chatml(batch):
    formatted = []
    for messages in batch["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted.append(text)
    return {"text": formatted}

raw_dataset = load_dataset("json", data_files={"train": TRAIN_DATA, "validation": VAL_DATA})
dataset = raw_dataset.map(format_chatml, batched=True)

print(f"Dataset loaded: {len(dataset['train'])} train examples, {len(dataset['validation'])} val examples.", flush=True)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_length=4096,
    packing=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=100,
    max_steps=350,
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

print("5. Launching SFT Training on NVIDIA A100...", flush=True)
train_result = trainer.train()

print("6. Saving Final SFT LoRA Adapter to Google Drive...", flush=True)
trainer.model.save_pretrained(FINAL_LORA_DIR)
tokenizer.save_pretrained(FINAL_LORA_DIR)

metrics = train_result.metrics
with open(os.path.join(FINAL_LORA_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"=== SFT Complete! Saved to {FINAL_LORA_DIR} ===", flush=True)
