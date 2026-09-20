# Autonomous Closed-Loop Agentic Coding Evaluation

## Executive Summary & Overview
This head-to-head evaluation assesses **Fable-Coder V4 (7.6B Replay-LoRA)** against its foundation base model **Qwen2.5-Coder-7B-Instruct** in an autonomous, closed-loop agentic workflow adhering to OpenCode agent architecture standards.

Unlike static single-turn generation benchmarks, this paradigm places each model inside an enclosed, secure containment development sandbox (`target_repo`) with live terminal tool access (`read_file`, `write_file`, and sandboxed `run_command`). The models must iteratively diagnose bugs, formulate multi-file patches, execute unit tests, self-correct upon failure, and verify final code integrity.

---

## Benchmark Highlights & Scoreboard

| Evaluation Dimension | Fable-Coder V4 (7.6B) | Qwen2.5-Coder-7B-Instruct | Advantage |
| :--- | :---: | :---: | :---: |
| **Task Pass Rate** | **25.0%** | **25.0%** | + 0.0% |
| **Average Turns to Resolution** | **3.25** | **3.75** | Fable faster |
| **Total Valid Diff Lines Generated** | **255** | **224** | Conciseness & Precision |
| **Total Evaluation Latency** | **71.07s** | **131.65s** | Local RTX 3060 CUDA |

---

## Detailed Case-by-Case Breakdown

### Case 1: Token Bucket Rate Limiter (Algorithms & Concurrency)
- **Target Code**: `src/rate_limiter.py`
- **Test Suite**: `tests/test_rate_limiter.py`
- **Task Objective**: The TokenBucketRateLimiter currently always returns False and is not thread-safe. Implement the token bucket algorithm in src/rate_limiter.py with thread safety using a lock, burst capacity allowance, and time-based refill based on refill_rate. Ensure allow_request(tokens) properly deducts tokens and refills them as time advances.

| Model | Status | Turns Taken | Execution Time | Diff Lines |
| :--- | :---: | :---: | :---: | :---: |
| **Fable-Coder V4** | PASS | 1 | 5.49s | 44 |
| **Qwen2.5-Coder-7B (Base)** | FAIL | 4 | 39.81s | 74 |

### Case 2: Cache Concurrency Bug (Runtime Mutation Error)
- **Target Code**: `src/cache_service.py`
- **Test Suite**: `tests/test_cache_service.py`
- **Task Objective**: In src/cache_service.py, clean_expired() crashes with 'RuntimeError: dictionary changed size during iteration' when removing expired keys. Fix clean_expired() so it cleanly and thread-safely evicts expired entries without mutating the dictionary while iterating over it.

| Model | Status | Turns Taken | Execution Time | Diff Lines |
| :--- | :---: | :---: | :---: | :---: |
| **Fable-Coder V4** | FAIL | 4 | 27.42s | 78 |
| **Qwen2.5-Coder-7B (Base)** | FAIL | 4 | 33.07s | 129 |

### Case 3: Cybersecurity & Red-Team Defense (SQLi & Injection)
- **Target Code**: `src/api_gateway.py`
- **Test Suite**: `tests/test_security.py`
- **Task Objective**: Fix two security vulnerabilities in src/api_gateway.py: 1) authenticate_user() concatenates user input directly into SQL strings, allowing SQL injection bypass. Refactor it to use parameterized queries (using ? placeholders). 2) sanitize_untrusted_note() fails to strip or neutralize prompt injection commands. Scrub or neutralize occurrences of 'SYSTEM OVERRIDE', 'IGNORE PREVIOUS INSTRUCTIONS', and similar override vectors.

| Model | Status | Turns Taken | Execution Time | Diff Lines |
| :--- | :---: | :---: | :---: | :---: |
| **Fable-Coder V4** | FAIL | 4 | 13.32s | 76 |
| **Qwen2.5-Coder-7B (Base)** | PASS | 3 | 15.02s | 21 |

### Case 4: Distributed Saga State Machine (Complex Architecture)
- **Target Code**: `src/order_saga.py`
- **Test Suite**: `tests/test_order_saga.py`
- **Task Objective**: Implement execute() in src/order_saga.py. Run steps in order. If any step fails, catch the error, roll back executed steps in reverse order (LIFO) calling step.compensate(ctx), mark compensated steps as COMPENSATED (or COMPENSATION_FAILED if compensation raises an error, recording compensation errors in SagaExecutionError), and raise SagaExecutionError.

| Model | Status | Turns Taken | Execution Time | Diff Lines |
| :--- | :---: | :---: | :---: | :---: |
| **Fable-Coder V4** | FAIL | 4 | 24.84s | 57 |
| **Qwen2.5-Coder-7B (Base)** | FAIL | 4 | 43.75s | 0 |

## Security & Containment Architecture
The evaluation was executed strictly inside an enclosed sandbox:
1. **Path Boundary Validation**: Every file read/write was validated against directory traversal escapes.
2. **Execution Guardrails**: Only whitelisted `pytest`, `python`, and `git` commands were permitted. Dangerous host-escaping commands (`powershell`, `rmdir /s`, `format`, socket creation) were intercepted and neutralized.
3. **Pristine State Guarantee**: `git reset --hard HEAD` and `git clean -fdx` were executed prior to every case run to guarantee zero residual cross-case pollution.
