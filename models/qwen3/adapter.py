"""Adapter for Qwen3: which GGUF files the pinned llama.cpp build runs.

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

SIZES = ["0.6b-q8_0", "8b-q4_k_m"]  # one row per GGUF file: <parameters>-<quant>, lowercase

# Qwen's own GGUF releases. Their licenses come from the model card.
WEIGHTS = {
    "0.6b-q8_0": {
        "repo": "Qwen/Qwen3-0.6B-GGUF",
        "revision": "23749fefcc72300e3a2ad315e1317431b06b590a",
        "files": ["Qwen3-0.6B-Q8_0.gguf"],
        "quant": "Q8_0",
    },
    "8b-q4_k_m": {
        "repo": "Qwen/Qwen3-8B-GGUF",
        "revision": "7c41481f57cb95916b40956ab2f0b139b296d974",
        "files": ["Qwen3-8B-Q4_K_M.gguf"],
        "quant": "Q4_K_M",
    },
}

# Optional: llama-server flags the official model card requires, such as "--jinja". The tool refuses
# the flags the protocol sets itself (context, GPU layers, batch sizes, cache types, slots).
# SERVER_ARGS = []

# The Qwen3 model card switches thinking off with enable_thinking=False in the chat template.
THINKING_OFF_ARGS = ["--chat-template-kwargs", '{"enable_thinking": false}']
