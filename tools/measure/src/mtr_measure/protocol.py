"""The measurement protocol's constants (docs/protocol.md). Adapters can't change them.

Every record carries PROTOCOL_VERSION. Raise it whenever a constant here changes meaning, so numbers
taken under different rules are never compared as if they were the same. Records without a
`protocol` field were taken under version 1: one run of 10 warm-up and 100 timed calls.
"""

from __future__ import annotations

PROTOCOL_VERSION = 2

# Image models (kind "image"): each run loads the model in a fresh worker process.
WARMUP = 10
TIMED = 100
RUNS = 3
# Largest allowed (max - min) / median of the runs' medians: 10%, set on 2026-10-26 from the first
# batch on the reference machine. Honest spreads were 0.2-6.8% (8.5% once), and 45% when the machine
# changed state in the middle of a run.
MAX_SPREAD = 0.10

# `mtr check`: enough calls to see that the adapter works, not a measurement.
CHECK_WARMUP = 3
CHECK_TIMED = 5

# CPU rows (runtimes pytorch-cpu, onnxruntime-cpu and llamacpp-cpu, docs/protocol.md "CPU rows"): the same three
# fresh processes and MAX_SPREAD, fewer calls, fp32 only, on CPU_THREADS threads (the physical cores, as PyTorch,
# ONNX Runtime and llama.cpp each default to). A call takes 30-1000 times longer than on the GPU, so each run makes
# CPU_WARMUP + CPU_TIMED calls, or the slow tier's when one call takes more than CPU_SLOW_CALL_S.
CPU_WARMUP = 5
CPU_TIMED = 30
CPU_SLOW_CALL_S = 10.0
CPU_SLOW_WARMUP = 1
CPU_SLOW_TIMED = 5
CPU_GEN_WARMUP = 1
CPU_GEN_TIMED = 5
CPU_LLM_WARMUP = 1
CPU_LLM_TIMED = 5
CPU_CHECK_WARMUP = 1
CPU_CHECK_TIMED = 2
# The longest call a CPU row times. A checkpoint whose call runs past it gets a record that says so instead of a number.
CPU_CALL_LIMIT_S = 300.0
# Warm-up goes on until it has lasted this long, so a desktop CPU's short boost above its sustained power limit is over
# before the timed calls start. Provisional: set from the first batch, as MAX_SPREAD was.
CPU_WARMUP_MIN_S = 60.0
# Largest allowed |median of the first half of the timed calls - median of the second half| / median. A run whose
# calls drift (a clock or power state changing) is refused instead of averaged.
CPU_STEADY_LIMIT = 0.05

# Accuracy evals pause nothing, so they need this much free VRAM on top of
# what `check` saw the model use.
EVAL_VRAM_MARGIN_MIB = 1024
# A GenEval run with --pause stops the machine's pause hooks (machine.toml) while it makes images, so the
# programs the margin above leaves room for aren't running. It still leaves this much for the desktop itself.
# Neither margin changes a score: it only decides whether the run starts.
EVAL_VRAM_MARGIN_PAUSED_MIB = 512

# Text-to-image models (kind "generation"): one call makes one image from the task's timing prompt, at the
# official resolution, steps and guidance, with a fixed seed. A call takes seconds, so each run makes fewer calls
# than an image model's, as a language model's do; three runs and MAX_SPREAD on the median apply as for image models.
GEN_WARMUP = 2
GEN_TIMED = 10
GEN_CHECK_WARMUP = 1
GEN_CHECK_TIMED = 2
GEN_SEED = 0
# Text-to-image accuracy: GenEval (https://github.com/djghosh13/geneval, MIT), its own evaluator at a pinned
# commit in its own environment (tools/measure/geneval/: Python 3.9, torch 1.12.1 with CUDA 11.3, mmdetection
# 2.28.2), as its README sets it up. 553 prompts, four images each, made with the adapter's GENERATION settings
# and nothing else: no prompt rewriter, no negative prompt. GenEval's own script seeds 42 at each prompt and
# draws the four images in a row; each of ours is one call with its own generator, seeded 42, 43, 44 and 45.
# The score is the mean of the six tasks' shares of correct images, 0-1 as GenEval reports it. Changing any of
# this means scoring every text-to-image model again.
GEN_BENCHMARKS = {
    "geneval": {"name": "GenEval", "metric": "overall: mean of the six tasks' accuracy", "images_per_prompt": 4,
                "seeds": [42, 43, 44, 45]},
}
GENEVAL_COMMIT = "af4902f24d3ca90ebbb446dd9891a59e0f82725f"

# Language models (kind "llm"): the pinned llama.cpp build (engine.py), all layers on the GPU,
# one request at a time. VRAM is reported at LLM_CONTEXT tokens of context; speed for a prompt of
# LLM_PROMPT_TOKENS tokens (llm_prompt.txt) and LLM_GEN_TOKENS generated tokens. Each run starts a
# fresh server, as each image run starts a fresh worker.
LLM_CONTEXT = 8192
LLM_PROMPT_TOKENS = 512
LLM_GEN_TOKENS = 256
LLM_WARMUP = 2
LLM_TIMED = 10
# Log verbosity 4 makes llama-server say how many layers it put on the GPU. It adds lines while
# loading only; decode speed was the same as at the default (2026-10-26).
LLM_SERVER_ARGS = ["-c", str(LLM_CONTEXT), "-np", "1", "-ngl", "999", "--fit", "off", "--no-webui", "-lv", "4"]
LLM_CHECK_WARMUP = 1
LLM_CHECK_TIMED = 2
# A language model's CPU row (llamacpp-cpu): the same binary, the GPU hidden and no device (-dev none), so no layer and
# no prompt batch reaches it; -ngl 0 alone would still send a long prompt's batches to the GPU. The threads (-t, -tb)
# are CPU_THREADS', added when the server starts.
LLM_CPU_SERVER_ARGS = ["-c", str(LLM_CONTEXT), "-np", "1", "-ngl", "0", "-dev", "none", "--fit", "off", "--no-webui", "-lv", "4"]

# Language-model accuracy: lm-evaluation-harness (tools/measure/lmeval/) against the same
# llama-server build, through its chat endpoint with the model's own chat template. Only generative
# tasks work there: llama-server can't return prompt log-probabilities, so multiple-choice tasks
# are out. The evaluation server opens several slots for throughput, each with LLM_EVAL_SLOT_CONTEXT
# tokens; accuracy doesn't depend on how requests are batched.
# The suite was fixed on 2026-10-26 from timing Qwen3-8B Q4_K_M: about 11 minutes per 8B
# checkpoint with 8 slots. Changing it means scoring every language model again.
LLM_EVAL_SLOTS = 8
LLM_EVAL_SLOT_CONTEXT = 8192
# Thinking off for models that can switch it, the way each model card says (the adapter's
# THINKING_OFF_ARGS), so models with and without a thinking mode compare alike. llama-server's generic --reasoning-budget 0 doesn't do it: on
# Qwen3 it only closes the thinking block, and the model goes on reasoning in its answer (2026-10-26).
LLM_EVAL_THINKING = "off"
LLM_BENCHMARKS = {
    # IFEval: 541 prompts with verifiable instructions, scored by rules. Headline: strict prompt-level accuracy.
    "ifeval": {"task": "ifeval", "metric": "prompt_level_strict_acc", "name": "IFEval"},
    # GSM8K: 1,319 grade-school math problems, 8 worked examples as a chat, answer "The final answer is N".
    # Headline: the last number of the answer (flexible-extract), so a missed phrase doesn't count as wrong math.
    # The task sets no answer limit, and lm-eval's API default of 256 tokens cut off up to a quarter of the answers of models
    # that reason at length, so it takes IFEval's 1280. Changing it means scoring every language model again. Each eval record
    # states its limit, and speed and VRAM don't depend on it, so PROTOCOL_VERSION stays.
    "gsm8k-cot": {"task": "gsm8k_cot_llama", "metric": "exact_match_flexible-extract", "name": "GSM8K, 8-shot chain of thought",
                  "max_gen_toks": 1280},
    # Left out: MMLU-Pro takes about a day per 8B checkpoint here; GPQA is gated on Hugging Face and has
    # only 198 questions; HumanEval and MBPP run the model's code, which needs a sandbox first.
}
