"""Adapter for Qwen3.5: which GGUF files the pinned llama.cpp build runs.

A language model needs no code here. The tool starts llama-server with the protocol's settings
(tools/measure/src/mtr_measure/protocol.py) and times a fixed workload. You only name the files.
- `mtr check <id> --size <size>` builds llama.cpp once and downloads the files.
- `mtr queue add <id> --size <size>` queues the timed run.

Where the files come from:
- Prefer the GGUF that the model's own organization publishes on Hugging Face.
- If there is none, a third-party quantization may be used. Name its repository and say so in the
  pull request. Licenses still come from the original model card (CONTRIBUTING.md).
- Pin `revision` to the repository's commit: "sha" in https://huggingface.co/api/models/<repo>.
"""

KIND = "llm"
TASK = "chat"

SIZES = ["9b-q4_k_m"]  # one row per GGUF file: <parameters>-<quant>, lowercase

# Qwen publishes no GGUF of Qwen3.5-9B. This is the unsloth quantization, a third party. The license comes
# from Qwen's own model card.
WEIGHTS = {
    "9b-q4_k_m": {
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "revision": "3885219b6810b007914f3a7950a8d1b469d598a5",
        "files": ["Qwen3.5-9B-Q4_K_M.gguf"],
        "quant": "Q4_K_M",
    },
}

# Optional: llama-server flags the official model card requires, such as "--jinja". The tool refuses
# the flags the protocol sets itself (context, GPU layers, batch sizes, cache types, slots).
# SERVER_ARGS = []

# The Qwen3.5 model card switches thinking off with enable_thinking=False in the chat template.
THINKING_OFF_ARGS = ["--chat-template-kwargs", '{"enable_thinking": false}']
