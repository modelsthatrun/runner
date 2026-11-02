"""Adapter for Qwen3.8: which GGUF files the pinned llama.cpp build runs.

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

SIZES = ["27b-q4_k_m"]  # one row per GGUF file: <parameters>-<quant>, lowercase

# Qwen publishes no GGUF of Qwen3.8. This is the ggml-org quantization, a third party. The license comes
# from Qwen's own model card.
WEIGHTS = {
    "27b-q4_k_m": {
        "repo": "ggml-org/Qwen3.8-27B-GGUF",
        "revision": "a316ea527fde32c12487be7574e2db5ac2ad2a7e",
        "files": ["Qwen3.8-27B-Q4_K_M.gguf"],
        "quant": "Q4_K_M",
    },
}

# Optional: llama-server flags the official model card requires, such as "--jinja". The tool refuses
# the flags the protocol sets itself (context, GPU layers, batch sizes, cache types, slots).
# SERVER_ARGS = []

# The Qwen3.8 model card switches thinking off with enable_thinking=False in the chat template.
THINKING_OFF_ARGS = ["--chat-template-kwargs", '{"enable_thinking": false}']
