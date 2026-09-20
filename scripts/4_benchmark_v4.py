import torch
import os
import sys
import json
import re

os.environ["USE_TF"] = "0"
sys.modules["tensorflow"] = None

from transformers import AutoTokenizer, AutoModelForCausalLM

print("=== Phase 4: Executing Fable-Coder V4 Closed Benchmark ===", flush=True)

MODEL_DIR = "/content/fable_v4_merged_bf16"
SPEC_PATH = "/content/benchmark_spec.json"

if not os.path.exists(SPEC_PATH):
    for p in ["/content/drive/MyDrive/benchmark_spec.json", "/content/MalxLabs-Fable5_QwenCoder/benchmark_spec.json"]:
        if os.path.exists(p):
            SPEC_PATH = p
            break

print(f"Loading Benchmark Spec from: {SPEC_PATH}...", flush=True)
with open(SPEC_PATH, "r", encoding="utf-8") as f:
    spec = json.load(f)

print("Loading Model & Tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

tasks = spec.get("tasks", [])
results = []
passed_count = 0
total_count = len(tasks)
total_points = 0
earned_points = 0

cat_stats = {}

for task in tasks:
    t_id = task["id"]
    cat = task.get("category", "unknown")
    title = task.get("title", "")
    prompt = task.get("prompt", "")
    points = task.get("points", 1)
    kind = task.get("kind", "")
    total_points += points
    
    if cat not in cat_stats:
        cat_stats[cat] = {"passed": 0, "total": 0, "pts_earned": 0, "pts_total": 0}
    cat_stats[cat]["total"] += 1
    cat_stats[cat]["pts_total"] += points

    system_msg = spec.get("system", "You are a helpful assistant.")
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.01,
            top_p=0.95,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    passed = False
    if kind == "exact_text":
        expected = task.get("expected_text", "").strip()
        passed = (response == expected)
    elif kind == "exact_any":
        expected_texts = [e.lower().strip() for e in task.get("expected_texts", [])]
        passed = any(e in response.lower() for e in expected_texts)
    elif kind == "json_exact":
        expected_obj = task.get("expected", {})
        try:
            m = re.search(r'\{.*\}', response, re.DOTALL)
            if m:
                res_obj = json.loads(m.group(0))
                passed = all(res_obj.get(k) == v for k, v in expected_obj.items())
        except Exception:
            passed = False
    elif kind == "constraints":
        max_words = task.get("max_words", 100)
        must_not = task.get("must_not", [])
        words = response.split()
        word_count_ok = (len(words) <= max_words)
        must_not_ok = not any(b.lower() in response.lower() for b in must_not)
        passed = (word_count_ok and must_not_ok)
    elif kind == "contains_all":
        req = task.get("required", [])
        forb = task.get("forbidden", [])
        has_req = all(r.lower() in response.lower() for r in req)
        no_forb = not any(f.lower() in response.lower() for f in forb)
        passed = (has_req and no_forb)
    elif kind in ["code_static", "sql_static"]:
        checks = task.get("code_checks", [])
        passed = all(c.lower() in response.lower() for c in checks)
    else:
        passed = len(response) > 0

    if passed:
        passed_count += 1
        earned_points += points
        cat_stats[cat]["passed"] += 1
        cat_stats[cat]["pts_earned"] += points

    status_str = "PASS" if passed else "FAIL"
    print(f"[{status_str}] Task {t_id} ({cat}): {title} [{points} pts]", flush=True)

    results.append({
        "id": t_id,
        "category": cat,
        "title": title,
        "passed": passed,
        "points_earned": points if passed else 0,
        "points_possible": points,
        "response_preview": response[:200]
    })

accuracy = (passed_count / total_count) * 100
weighted_score = (earned_points / total_points) * 100

print(f"\nTotal Tasks Passed: {passed_count}/{total_count} ({accuracy:.1f}%)", flush=True)
print(f"Weighted Score:     {earned_points}/{total_points} ({weighted_score:.1f}%)\n", flush=True)

report = {
    "model": "Fable-Coder-V4-7B",
    "architecture": "Replay-Anchored LoRA (Rank 32, Alpha 64, All 7 Projections)",
    "accuracy_pct": accuracy,
    "weighted_score_pct": weighted_score,
    "tasks_passed": passed_count,
    "total_tasks": total_count,
    "category_breakdown": cat_stats,
    "results": results
}

DRIVE_REPORT = "/content/drive/MyDrive/Fable_Coder_V4/BENCHMARK_RESULTS_V4.json"
with open(DRIVE_REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"\nSaved benchmark report to Google Drive: {DRIVE_REPORT}", flush=True)
