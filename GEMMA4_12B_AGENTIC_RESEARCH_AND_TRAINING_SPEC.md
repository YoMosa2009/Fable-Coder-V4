# Comprehensive Technical Research & Training Specification: Gemma-4-12B-it Agentic Super-Base

**Document Version:** 1.0.0  
**Target Foundation Model:** `google/gemma-4-12B-it`  
**Target Hardware:** Google Colab (Single A100-SXM4 40GB / 80GB GPU, 24–48 Hour Budget)  
**Primary Focus:** Closed-Loop Multi-Turn Agentic Tool Use, Complex Software Engineering, Autonomous Error Recovery, Defensive Cybersecurity, and Long-Horizon System Architecture.

---

## 1. Executive Summary & Core Mission

Across four iterations of the Fable-Coder series (V1 through V4), substantial progress was made in understanding fine-tuning dynamics, loss masking, replay buffers, and benchmark evaluation. However, empirical evaluation on closed-loop execution sandboxes reveals a fundamental bottleneck: **smaller 7B models trained primarily on single-turn synthetic instruction pairs lack the inductive capacity, thinking depth, and multi-turn error-recovery resilience needed to operate reliably as autonomous agents**.

This project establishes the end-to-end blueprint to train a new model built upon Google DeepMind's **`google/gemma-4-12B-it`**. Released in mid-2026, Gemma 4 12B features a native encoder-free multimodal architecture, a 256K token context window, native system instruction adherence, and dedicated **Thinking Channels** (`<|channel>thought\n...<channel|>`) and **Agentic Tool Protocols** (`<|tool_call|>`, `<|tool_response|>`).

### The Golden Objective
The fine-tuned model must **decisively outperform the base `google/gemma-4-12B-it`** across both static multi-domain benchmarks (OmniAgent-Bench) and real-world multi-turn execution sandboxes (terminal commands, test suites, multi-file refactoring, vulnerability remediation), with **zero regression** in general instruction following or mathematical reasoning.

---

## 2. Empirical Post-Mortem: What Went Wrong in V1–V4 and How We Prevent It

To guarantee success, every systemic flaw observed from V1 to V4 must be explicitly diagnosed and eradicated.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FABLE-CODER ITERATIVE EVOLUTION                       │
├─────────────┬─────────────────┬─────────────────────────────────────────────┤
│ Iteration   │ Benchmark Score │ Primary Failure Mode / Post-Mortem Cause     │
├─────────────┼─────────────────┼─────────────────────────────────────────────┤
│ V1 (Base)   │ 60.0%           │ Naive SFT + simple DPO degraded planning.   │
│ V2 (Data)   │ 43.3%           │ Toxic synthetic data corrupted reasoning.   │
│ V3 (DPO)    │ 50.0%           │ Catastrophic 6-pair DPO mode collapse.      │
│ V4 (Omni)   │ 72.69% (Static) │ Single-turn SFT failed closed-loop tests    │
│             │ 25.0% (Sandbox) │ (1/4 pass: broke on runtime dict mutation,  │
│             │                 │ Saga compensation, and SQLi multi-file patch)│
└─────────────┴─────────────────┴─────────────────────────────────────────────┘
```

### Detailed Failure Mode Analysis

#### 1. Fable-Coder V1: The Naive SFT Pitfall
- **Observed Behavior:** Strong baseline syntax completion, but erratic tool calling and severe hallucinations on complex distributed architectures.
- **Root Cause:** Standard SFT with short context lengths (2048 tokens) without loss masking on system and user prompts taught the model to memorize prompt formatting rather than learning conditional response generation.

#### 2. Fable-Coder V2: Synthetic Data Poisoning
- **Observed Behavior:** Performance collapsed from 60.0% to 43.3%; bug-fixing capability dropped to 16.7%.
- **Root Cause:** Unfiltered ingestion of low-quality, synthetically generated code pairs containing subtle logical fallacies, unhandled exceptions, and dead code. The model learned bad coding habits and lost confidence in standard algorithmic patterns.

#### 3. Fable-Coder V3: Catastrophic Few-Pair DPO Collapse
- **Observed Behavior:** Algorithm score dropped from 5/6 (83.3%) to 3/6 (50.0%). Code generations became truncated, and intermediate reasoning vanished.
- **Root Cause:** Applying Direct Preference Optimization (DPO) over an ultra-small batch of only 6 paired samples repeated across 80 optimization steps with length penalty biases. The DPO loss heavily penalized token length, causing the policy to violently collapse into minimal, clipped outputs that omitted edge-case checks and architectural planning.

#### 4. Fable-Coder V4: The Single-Turn Illusion & Multi-Turn Sandbox Failure
- **Observed Behavior:** V4 achieved 72.69% on the static OmniAgent benchmark (outscoring base at 70.72%), passing 100% of single-turn tool, cybersecurity, and long-horizon questions. **However, in the real-world autonomous closed-loop sandbox (`eval/AGENTIC_EVAL_REPORT.md`), it scored only 25.0% (1/4 tasks passed)**:
  - *Case 1 (Rate Limiter):* Passed in Turn 1 (5.49s). Single-turn algorithmic pattern matched.
  - *Case 2 (Cache Concurrency):* Failed after 5 turns. The model mutated a Python dictionary during iteration (`RuntimeError: dictionary changed size during iteration`), failed to understand the traceback returned by `pytest`, and entered a repetitive editing loop.
  - *Case 3 (Security Defense):* Failed. The model fixed only the SQL injection in the primary handler but failed to neutralize the nested prompt injection attack vector in the user audit logger, resulting in unit test failure.
  - *Case 4 (Distributed Saga):* Failed. The model wrote compensation stubs but inverted the reverse-order rollback logic when step 3 failed.
- **Root Cause:** **V4 was trained on isolated static turns.** It had never been trained on real *execution feedback loops*—receiving a terminal error or failing test trace, reasoning about the error in an internal thinking block, and outputting a targeted patch.

---

## 3. Foundation Architecture: `google/gemma-4-12B-it`

### 3.1 Architectural Profile
- **Developer:** Google DeepMind (Released June 2026).
- **Parameter Count:** 12 Billion dense parameters.
- **Architecture Type:** Decoder-only, **Encoder-Free Multimodal** (projects raw visual patches and audio waveforms directly into the decoder's embedding space without separate bottleneck encoders).
- **Context Window:** Up to 256,000 tokens (we will train on 8,192 tokens packing).
- **Vocabulary Size:** ~256,000 tokens (extended multimodal vocabulary).
- **License:** Apache 2.0 (fully permissive commercial and research use).

### 3.2 Gemma 4 Chat & Thinking Template Specification
Unlike Gemma 1, 2, and 3 which used `<start_of_turn>` and `<end_of_turn>`, **Gemma 4 introduces a structured, channel-aware turn template**:

```jinja
<|turn>system
{{ system_prompt }}<turn|>
<|turn>user
{{ user_message }}<turn|>
<|turn>model
<|channel>thought
{{ internal_chain_of_thought_reasoning }}<channel|>
<|channel>call:{{ tool_name }}{{ tool_args_json }}<channel|>
{{ response_content }}<turn|>
```

#### Key Special Tokens & Roles
1. `<|turn>role\n...<turn|>`: Delimits each conversation turn (roles: `system`, `user`, `model`).
2. `<|channel>thought\n...<channel|>`: Dedicated thinking channel. Tokens generated inside this block are treated as private reasoning traces. The model learns to break down requirements, analyze failure logs, and verify logic before emitting code or tool invocations.
3. `<|tool_call|>` / `<|channel>call:tool_name{...}<channel|>`: Standardized tool execution request.
4. `<|tool_response|>` / `<|turn>tool\n...<turn|>`: Structured environment execution response containing file contents, compiler stderr, or test results.

---

## 4. Curated Multi-Pillar Training Dataset Specification

To avoid the data traps of V2 and V3, the training mixture uses a strict **"Less-is-More" STITCH (Sliding-memory Trajectory Inference and Task Chunking)** paradigm: **8,000 to 12,000 high-density, multi-turn, verified trajectories**, strictly balanced across 5 pillars.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 GEMMA-4-12B-IT DATASET CURATION (10,000 SAMPLES)            │
├───────────────┬────────┬────────────────────────────────────────────────────┤
│ Pillar Name   │ Share  │ Source & Focus                                     │
├───────────────┼────────┼────────────────────────────────────────────────────┤
│ 1. Agentic    │ 25%    │ Nanbeige/ToolMind & NVIDIA Nemotron-Agentic-v1     │
│    Tool Use   │ (2.5k) │ Multi-step tool dispatch, file I/O, terminal cmds  │
├───────────────┼────────┼────────────────────────────────────────────────────┤
│ 2. Closed-Loop│ 30%    │ SWE-bench Verified trajectories & DeepSWE-Preview   │
│    SWE & Fix  │ (3.0k) │ Compiler stderr / pytest trace -> diagnose -> fix  │
├───────────────┼────────┼────────────────────────────────────────────────────┤
│ 3. Defensive  │ 20%    │ Trendyol Cybersecurity v2 & pentesting-explanations│
│    Security   │ (2.0k) │ Vulnerability remediation, prompt injection defense│
├───────────────┼────────┼────────────────────────────────────────────────────┤
│ 4. Long-      │ 15%    │ System architecture, distributed consensus, Sagas, │
│    Horizon Sys│ (1.5k) │ thread concurrency, lock-free queues + thought trace│
├───────────────┼────────┼────────────────────────────────────────────────────┤
│ 5. Foundation │ 10%    │ General instruction anchor, logic & algorithmic CoT│
│    Anchor     │ (1.0k) │ Prevents catastrophic forgetting & policy drift    │
└───────────────┴────────┴────────────────────────────────────────────────────┘
```

### Pillar Breakdown & Quality Filtering

#### Pillar 1: Agentic Tool Use & Environment Interaction (25% / 2,500 samples)
- **Datasets:** `Nanbeige/ToolMind`, `nvidia/Nemotron-Agentic-v1`.
- **Filtering Standard:** Multi-turn sequences where user provides high-level objectives. The model must call `read_file`, `write_file`, `run_command`, or `grep_search`. Every tool call must be preceded by a `<|channel>thought` block explaining *why* the tool is being invoked and *what output is expected*.

#### Pillar 2: Closed-Loop Software Engineering & Self-Correction (30% / 3,000 samples)
- **Datasets:** `SWE-bench_Verified` filtered trajectories, `agentica-org/DeepSWE-Preview`, `CodeFeedback-Filtered-Instruction`.
- **Filtering Standard:** **Crucial anti-V4 countermeasure.** Samples MUST feature error recovery loops:
  - Turn 1: Model inspects codebase and proposes fix.
  - Turn 2 (Environment): `pytest` output showing `AssertionError` or `RuntimeError: dictionary changed size during iteration`.
  - Turn 3 (Model): Model enters `<|channel>thought`, analyzes the stack trace line by line, identifies the root cause (e.g. iterating over `dict.keys()` without a shallow copy), and applies the correct targeted patch.

#### Pillar 3: Defensive Cybersecurity & Vulnerability Remediation (20% / 2,000 samples)
- **Datasets:** `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset` (v2.0), `theelderemo/pentesting-explanations` (filtered strictly for defensive remediation).
- **Focus:**
  - SQL injection parameterization (prepared statements, ORM secure binding).
  - Cross-Site Scripting (context-aware HTML/JS escaping).
  - Server-Side Request Forgery (SSRF) validation (strict URL schema, private IP range blocks).
  - **Indirect Prompt Injection Neutralization:** Sanitizing and framing untrusted third-party inputs before passing them into downstream LLM calls.

#### Pillar 4: Long-Horizon System Architecture & Concurrency (15% / 1,500 samples)
- **Focus:** Multi-threaded synchronization, lock-free queues, read-write locks, distributed 2-Phase Commit and Saga patterns with compensatory rollbacks, sliding-window rate limiters.
- **Requirement:** Comprehensive thinking blocks that diagram invariants and state transitions before code generation.

#### Pillar 5: Core General Capabilities & Anchor Replay Buffer (10% / 1,000 samples)
- **Datasets:** Ultra-filtered subsets of OpenHermes-2.5 and GSM8K/MATH CoT.
- **Function:** Ensures zero degradation in conversational coherence, tool-unrelated coding tasks, or basic mathematical reasoning.

---

## 5. Google Colab A100 Training Pipeline & Hyperparameters

### 5.1 Hardware Constraints & Strategy
- **Platform:** Google Colab Pro / Pro+
- **Compute Resource:** Single NVIDIA A100-SXM4 (40GB or 80GB VRAM)
- **Target Wall-Clock Training Duration:** 20 to 36 hours (fits within 1–2 days budget).
- **Framework:** PyTorch 2.4+, Hugging Face `transformers`, `peft`, `trl` (`SFTTrainer`), and `bitsandbytes`.

### 5.2 Memory Optimization Matrix (12B on A100 40GB)

```
┌──────────────────────────────────────┬─────────────────────────────────────┐
│ Component                            │ VRAM Allocation                     │
├──────────────────────────────────────┼─────────────────────────────────────┤
│ Base Model (12B in 4-bit NF4)        │ ~7.2 GB                             │
│ LoRA Adapters (r=64, all linear)     │ ~1.6 GB                             │
│ Optimizer States (Paged AdamW 8-bit) │ ~1.8 GB                             │
│ Activations (Seq Len 8192 + Pack)    │ ~18.5 GB (with FlashAttention-2)    │
│ Gradient Checkpointing Overhead      │ ~2.5 GB                             │
│ CUDA Workspace Buffer                │ ~3.0 GB                             │
├──────────────────────────────────────┼─────────────────────────────────────┤
│ Total Peak VRAM Usage                │ ~34.6 GB / 40.0 GB (Safe margin!)   │
└──────────────────────────────────────┴─────────────────────────────────────┘
```

### 5.3 Concrete Hyperparameter Configuration

```python
# Training Configuration for Gemma-4-12B-it Agentic QLoRA
TRAINING_CONFIG = {
    # Foundation Model
    "model_id": "google/gemma-4-12B-it",
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",  # Falls back to sdpa
    
    # QLoRA Quantization (bitsandbytes)
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    
    # PEFT / LoRA Parameters
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
    
    # Optimization & Scheduling
    "optimizer": "paged_adamw_8bit",
    "learning_rate": 1.2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.04,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    
    # Batch Sizing & Throughput (Effective Batch Size = 32)
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "max_seq_length": 8192,
    "packing": True,  # Constant memory utilization, no wasted pad tokens
    "gradient_checkpointing": True,
    
    # Training Budget
    "num_train_epochs": 2,
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 3,
    
    # Loss Masking Strategy
    # ONLY calculate cross-entropy loss on assistant tokens and thinking blocks!
    "mask_user_and_system_loss": True,
}
```

---

## 6. The 5 Inviolable Rules (Anti-Regression Protocol)

To permanently prevent the regressions of V1 through V4:

1. **NEVER Run DPO on Small Synthetic Sets:** V3 collapsed because 6 pairs were optimized for 80 steps. If DPO or KTO is ever applied, it must contain a minimum of 2,500 diverse, length-normalized pairs with human or unit-test ground truth.
2. **NEVER Suppress Thinking Channels:** The `<|channel>thought` tokens must never be penalized with negative length biases. The reasoning trace is the primary engine of multi-step problem solving.
3. **NEVER Train Without Loss Masking:** Compute loss exclusively on assistant tokens (reasoning, tool calls, and final responses). Computing loss on user prompts or environment execution outputs causes the model to mimic user questions and confuse environment outputs with its own generations.
4. **NEVER Bypass Multi-Turn Error Recovery Trajectories:** At least 30% of the training corpus must contain execution failures (syntax errors, test failures, runtime exceptions) followed by diagnosis and successful refactoring.
5. **NEVER Alter Native Special Tokens:** Do not invent non-standard tokens that conflict with Gemma 4's tokenizer (`<|turn>`, `<turn|>`, `<|channel>thought`, `<channel|>`). Always utilize Hugging Face's official `apply_chat_template` or the exact Jinja specification.

---

## 7. Comprehensive Verification & Evaluation Plan

### 7.1 Stage 1: Static Multi-Domain Benchmark (OmniAgent-Bench)
The fine-tuned model and base `google/gemma-4-12B-it` will be evaluated head-to-head on the 16-task universal benchmark:
- **Agentic Tools (3 tasks):** JSON schema tool dispatch, multi-step pipeline composition, parameter validation.
- **Cybersecurity (3 tasks):** SQLi/XSS/SSRF remediation, secure password hashing, prompt injection defense.
- **Long-Horizon System Architecture (2 tasks):** Distributed Saga state machine with reverse compensation, thread-safe concurrent cache with LRU eviction.
- **Coding & Algorithms (5 tasks):** LeetCode hard dynamic programming, tree traversals, graph algorithms.
- **Multi-Turn Reasoning (3 tasks):** State tracking across 4 conversational turns.

### 7.2 Stage 2: Autonomous Closed-Loop Sandbox Evaluation
Using the OpenCode agentic runner architecture (`read_file`, `write_file`, `run_command`, `pytest`):
- **Case 1: Token Bucket Rate Limiter (Thread Safety & Drift Control)**
- **Case 2: Concurrent LRU Cache (Runtime Dict Mutation & Lock Contention)**
- **Case 3: Dual Security Patch (SQL Injection Binding + Audit Log Prompt Injection Neutralization)**
- **Case 4: Distributed Saga State Machine (Compensation Inversion & Failure Rollback)**
- **Pass Criterion:** The model must achieve $\ge 75.0\%$ (at least 3/4 tasks passed) on closed-loop execution, directly solving the failure modes of Fable-Coder V4.

### 7.3 Stage 3: GGUF Quantization & Deployment
Following fine-tuning, the adapter weights will be merged into 16-bit FP16/BF16 base weights and exported using `llama.cpp` to high-performance quantized GGUF formats:
- `Q4_K_M`: Optimal for local 16GB RAM / 8GB VRAM execution.
- `Q5_K_M`: Near-lossless precision for 24GB VRAM workstation inference.
- `Q8_0`: Reference quantized baseline.
- `Modelfile` configurations for Ollama, LM Studio, and OpenCode CLI.

---

## 8. Implementation Roadmap for Google Colab

| Phase | Milestone | Expected Duration | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Colab Environment Setup & Dependencies | 1 Hour | PyTorch 2.4, FlashAttention-2, bitsandbytes, TRL, HuggingFace auth |
| **Phase 2** | Dataset Ingestion, Formatting & Masking | 2–3 Hours | 10k tokenized & packed sequences with thinking channels & loss masking |
| **Phase 3** | Gemma-4-12B-it QLoRA SFT Execution | 20–28 Hours | 2 Epochs on A100 GPU, continuous checkpointing to Google Drive |
| **Phase 4** | Adapter Merge, Verification & Benchmarking | 3–4 Hours | OmniAgent-Bench & Closed-loop sandbox evaluation vs. Base Gemma 4 |
| **Phase 5** | GGUF Quantization & Repository Release | 2 Hours | Export `Q4_K_M` GGUF, push model cards to Hugging Face & GitHub |

---
*Authored by Antigravity Engineering for Google DeepMind & YoMosa2009.*
