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

# Accuracy evals pause nothing, so they need this much free VRAM on top of
# what `check` saw the model use.
EVAL_VRAM_MARGIN_MIB = 1024

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
    "gsm8k-cot": {"task": "gsm8k_cot_llama", "metric": "exact_match_flexible-extract", "name": "GSM8K, 8-shot chain of thought"},
    # Left out: MMLU-Pro takes about a day per 8B checkpoint here; GPQA is gated on Hugging Face and has
    # only 198 questions; HumanEval and MBPP run the model's code, which needs a sandbox first.
}
