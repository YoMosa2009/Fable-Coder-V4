# Fable-Coder V4: 7.6B Generalized Agentic & CyberSec Code LLM

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model Size: 7.6B](https://img.shields.io/badge/Parameters-7.6B-green.svg)](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
[![Format: GGUF Q4_K_M](https://img.shields.io/badge/Format-GGUF%20Q4__K__M-orange.svg)](https://github.com/ggerganov/llama.cpp)
[![Hardware: NVIDIA A100-SXM4-40GB](https://img.shields.io/badge/Hardware-NVIDIA%20A100--SXM4--40GB-76B900.svg)](https://www.nvidia.com/en-us/data-center/a100/)

**Fable-Coder V4** is a generalized, high-intelligence 7.6B code and agentic reasoning model engineered to decisively surpass baseline capabilities across coding, bug fixing, tool calling, cybersecurity, and multi-step reasoning.

---

## 1. Architectural Evolution & Benchmark Trajectory

| Model Version | Overall Score | Pass Rate | Coding & Algo | Bug Fixing | Agentic Tools | Math & Bayesian | Instruction & Sec | Architecture & Methodology |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Qwen2.5-Coder-7B-Instruct (Base)** | 18/30 | 60.0% | 5/6 (83.3%) | 3/6 (50.0%) | 3/6 (50.0%) | 4/6 (66.7%) | 3/6 (50.0%) | Alibaba Cloud Pre-trained Baseline |
| **Fable-Coder V1** | 18/30 | 60.0% | 5/6 (83.3%) | 4/6 (66.7%) | 3/6 (50.0%) | 3/6 (50.0%) | 3/6 (50.0%) | LoRA SFT + DPO Alignment |
| **Fable-Coder V2** | 13/30 | 43.3% | 4/6 (66.7%) | 1/6 (16.7%) | 3/6 (50.0%) | 2/6 (33.3%) | 3/6 (50.0%) | Full Linear LoRA + Synthetic Traces (Regression) |
| **Fable-Coder V3** | 15/30 | 50.0% | 3/6 (50.0%) | 4/6 (66.7%) | 3/6 (50.0%) | 3/6 (50.0%) | 2/6 (33.3%) | Frozen MLP LoRA + 6-Pair Anti-Bloat DPO |
| **Fable-Coder V4 (Target)** | **$\ge$ 25/30** | **$\ge$ 83.3%** | **6/6 (100%)** | **5/6 (83.3%)** | **5/6 (83.3%)** | **5/6 (83.3%)** | **5/6 (83.3%)** | **4-Pillar Replay-Anchored LoRA (Rank 32, Alpha 64)** |

---

## 2. Root Cause Analysis: Why V3 Regressed and How V4 Solves It

1. **The 6-Pair DPO Mode Collapse**: In V3, DPO was trained on only 6 distinct pairs repeated 25 times over 80 optimizer steps. The severe penalty on reasoning tokens caused the model to suppress intermediate planning across all tasks, causing Coding & Algo to collapse from 5/6 to 3/6.
2. **Attention Drift Without Rehearsal**: Without a general coding replay buffer, updating attention projections degraded Qwen's foundational coding knowledge.
3. **The V4 Solution**:
   - **Start Fresh from Base**: Zero inherited mode collapse.
   - **4-Pillar Dataset Mix**:
     - 40% General Coding Replay (`m-a-p/CodeFeedback-Filtered-Instruction`)
     - 25% Agentic Tool Trajectories (`NousResearch/Hermes-Function-Calling`)
     - 20% Cybersecurity & Injection Resistance (`sh111111111111111/agentic_red_team`)
     - 15% Precision Instruction & Logic Constraints
   - **Conservative Hyperparameters**: Rank 32, Alpha 64 across all 7 projections, conservative learning rate ($2\times 10^{-5}$) with cosine schedule and loss masking.

---

## 3. Quickstart: Ollama & llama.cpp

### Run with Ollama
```bash
# Clone the repository
git clone https://github.com/YoMosa2009/Fable-Coder-V4.git
cd Fable-Coder-V4

# Place the downloaded GGUF binary in this directory
# Build and run
ollama create fable-coder-v4 -f ./Modelfile
ollama run fable-coder-v4
```

### Run with llama.cpp
```bash
./llama-cli -m fable-coder-v4-7b.Q4_K_M.gguf \
    -p "Write a Python function to safely validate path traversal without accessing the filesystem" \
    -c 4096 --temp 0.2
```

---

## 4. Deliverable Verification
- **GGUF File**: `fable-coder-v4-7b.Q4_K_M.gguf`
- **Location**: Google Drive (`/content/drive/MyDrive/Fable_Coder_V4/`)
- **Quantization**: `Q4_K_M` (k-quant Medium)
