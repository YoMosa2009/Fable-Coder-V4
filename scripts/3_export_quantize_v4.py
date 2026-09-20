import torch
import os
import sys
import subprocess
import hashlib
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("=== Phase 3: Full-Precision Merge & GGUF Quantization (V4) ===", flush=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf"], check=True)
print("Verified gguf python package.", flush=True)

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
SFT_LORA_DIR = "/content/drive/MyDrive/Fable_Coder_V4/fable_v4_lora_sft"
MERGED_DIR = "/content/fable_v4_merged_bf16"
GGUF_FP16 = "/content/fable-coder-v4-fp16.gguf"
GGUF_Q4 = "/content/fable-coder-v4-7b.Q4_K_M.gguf"
DRIVE_FINAL_DIR = "/content/drive/MyDrive/Fable_Coder_V4"
os.makedirs(DRIVE_FINAL_DIR, exist_ok=True)
DRIVE_FINAL_GGUF = f"{DRIVE_FINAL_DIR}/fable-coder-v4-7b.Q4_K_M.gguf"

print(f"1. Merging adapter weights from: {SFT_LORA_DIR}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

peft_model = PeftModel.from_pretrained(base_model, SFT_LORA_DIR)
print("Executing unquantized BF16 merge_and_unload()...", flush=True)
merged_model = peft_model.merge_and_unload()

print(f"Saving merged BF16 weights to {MERGED_DIR}...", flush=True)
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print("Merged weights saved successfully.", flush=True)

del merged_model, base_model, peft_model
gc.collect()
torch.cuda.empty_cache()

# Sanitize tokenizer_config.json
tok_cfg_path = os.path.join(MERGED_DIR, "tokenizer_config.json")
if os.path.exists(tok_cfg_path):
    with open(tok_cfg_path, "r", encoding="utf-8") as f:
        tok_cfg = json.load(f)
    if "extra_special_tokens" in tok_cfg and isinstance(tok_cfg["extra_special_tokens"], list):
        del tok_cfg["extra_special_tokens"]
        with open(tok_cfg_path, "w", encoding="utf-8") as f:
            json.dump(tok_cfg, f, indent=2)

print("2. Converting Hugging Face safetensors to FP16 GGUF...", flush=True)
convert_cmd = f"python /content/llama.cpp/convert_hf_to_gguf.py {MERGED_DIR} --outfile {GGUF_FP16} --outtype f16"
res_c = subprocess.run(convert_cmd, shell=True, capture_output=True, text=True)
print(res_c.stdout[-600:] if res_c.stdout else "")
if res_c.returncode != 0:
    print("Convert error stderr:", res_c.stderr)
    raise RuntimeError(f"convert_hf_to_gguf.py failed with code {res_c.returncode}")
else:
    print(f"FP16 GGUF created successfully: {os.path.getsize(GGUF_FP16) / (1024**3):.2f} GiB", flush=True)

print("3. Quantizing to Q4_K_M via llama-quantize...", flush=True)
quant_cmd = f"/content/llama.cpp/build-cpu/bin/llama-quantize {GGUF_FP16} {GGUF_Q4} Q4_K_M 8"
res_q = subprocess.run(quant_cmd, shell=True, capture_output=True, text=True)
print(res_q.stdout[-600:] if res_q.stdout else "")
if res_q.returncode != 0:
    print("Quantization error stderr:", res_q.stderr)
    raise RuntimeError(f"llama-quantize failed with code {res_q.returncode}")
else:
    q_size = os.path.getsize(GGUF_Q4)
    print(f"Quantized GGUF created: {q_size} bytes ({q_size / (1024**3):.2f} GiB)", flush=True)

print("4. Calculating SHA-256 Checksum...", flush=True)
sha256_hash = hashlib.sha256()
with open(GGUF_Q4, "rb") as f:
    for byte_block in iter(lambda: f.read(65536), b""):
        sha256_hash.update(byte_block)
checksum = sha256_hash.hexdigest()
print(f"SHA-256 Checksum: {checksum}", flush=True)

chk_path = f"{DRIVE_FINAL_DIR}/SHA256SUMS"
with open(chk_path, "w", encoding="utf-8") as f:
    f.write(f"{checksum}  fable-coder-v4-7b.Q4_K_M.gguf\n")

print("5. Persisting final GGUF binary to Google Drive...", flush=True)
cp_cmd = f"cp {GGUF_Q4} {DRIVE_FINAL_GGUF}"
subprocess.run(cp_cmd, shell=True, check=True)
print(f"SUCCESS: Binary saved to Google Drive at {DRIVE_FINAL_GGUF}", flush=True)

if os.path.exists(GGUF_FP16):
    os.remove(GGUF_FP16)

print("=== Phase 3 Merge & Quantization Complete ===", flush=True)
