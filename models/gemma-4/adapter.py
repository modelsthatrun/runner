"""Adapter for Gemma 4: which GGUF files the pinned llama.cpp build runs.

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

SIZES = ["12b-q4_0", "31b-q4_0"]  # one row per GGUF file: <parameters>-<quant>, lowercase

# Google's own GGUF releases, quantized with quantization-aware training. Their licenses come from the model card.
WEIGHTS = {
    "12b-q4_0": {
        "repo": "google/gemma-4-12B-it-qat-q4_0-gguf",
        "revision": "29d097773436b69ff9feafd636ab4cf873786537",
        "files": ["gemma-4-12b-it-qat-q4_0.gguf"],
        "quant": "Q4_0",
    },
    "31b-q4_0": {
        "repo": "google/gemma-4-31B-it-qat-q4_0-gguf",
        "revision": "59dde24573e7e61570dba08b18a2e1fe246955ed",
        "files": ["gemma-4-31B_q4_0-it.gguf"],
        "quant": "Q4_0",
    },
}

# Optional: llama-server flags the official model card requires, such as "--jinja". The tool refuses
# the flags the protocol sets itself (context, GPU layers, batch sizes, cache types, slots).
# SERVER_ARGS = []

# The Gemma 4 model card switches thinking off with enable_thinking=False in the chat template. The
# template then fills in the empty thought block itself, so the answer starts right after it.
THINKING_OFF_ARGS = ["--chat-template-kwargs", '{"enable_thinking": false}']
