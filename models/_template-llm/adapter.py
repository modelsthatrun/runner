"""Adapter for <model name> <size>: which GGUF files the pinned llama.cpp build runs.

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
TASK = "chat"  # the task it solves, as `mtr models` lists it

SIZES = ["8b-q4_k_m"]  # one row per GGUF file: <parameters>-<quant>, lowercase

WEIGHTS = {
    "8b-q4_k_m": {
        "repo": "org/Model-8B-GGUF",
        "revision": "0000000000000000000000000000000000000000",  # the 40-hex commit, never "main"
        "files": ["Model-8B-Q4_K_M.gguf"],  # split GGUF: every part, in order, first part first
        "quant": "Q4_K_M",
    },
}

# Optional: llama-server flags the official model card requires, such as "--jinja". The tool refuses
# the flags the protocol sets itself (context, GPU layers, batch sizes, cache types, slots).
# SERVER_ARGS = []

# For a model with a thinking mode: how its model card turns thinking off. `mtr eval` uses
# it, because accuracy is measured with thinking off for now (protocol.py, LLM_EVAL_THINKING).
# THINKING_OFF_ARGS = ["--chat-template-kwargs", '{"enable_thinking": false}']
