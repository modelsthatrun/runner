# Adapters

Every model family has one adapter, `models/<id>/adapter.py`. It tells the shared tool how to load a checkpoint and how to make one call from input to output, the way the model's official README does. The tool knows nothing else about the model.
- `mtr check`, `mtr run` and `mtr eval` use the adapter.
- `mtr outputs` uses the same adapter to draw the model's outputs.
- The interface is defined in `tools/measure/src/mtr_measure/adapter.py`.
- The starting points are `models/_template/` and `models/_template-llm/`, copied by `mtr new <id>`.

## Image models

```python
TASK: str                        # e.g. "object-detection"; picks the timing image and the prediction type
SIZES: list[str]                 # checkpoint sizes, e.g. ["nano", "small", "medium"]
PRECISIONS: list[str]            # what the official code supports: "fp16", "fp32", "bf16"
PACKAGE: str                     # the installed distribution that runs the model; its version is recorded
INPUT: dict[str, str]            # official default input resolution per size, "WxH"

def load(size: str, precision: str) -> Model: ...    # the checkpoint on the GPU at this precision
def predict(model: Model, x) -> object: ...          # one call, input to output, any output type
def weight_files(size: str) -> list[pathlib.Path]: ...   # the files load() read, main checkpoint first
def to_prediction(model, x, out): ...                # predict()'s output as the task's standard type
```

**Optional:**

| Name | Use it when |
|---|---|
| `prepare_input(paths)` | The model takes several files, or an input that isn't one RGB image. By default `x` is the input file decoded as an RGB PIL image |
| `input_size(size, x)` | The resolution depends on the input. Use it instead of `INPUT` |
| `package_version()` | The model runs from a git checkout rather than an installed package. Use it instead of `PACKAGE`, e.g. `"owner/repo@<commit>"` |
| `LATENCY_SCOPE` | One `predict()` call covers something other than pre-processing, forward and post-processing |
| `eval_predict(model, x)` | The official evaluation calls the model differently from its README, e.g. a lower score threshold. `mtr eval` uses it instead of `predict()` |
| `EVAL = {"label_map": {...}}` | The model's class names differ from the benchmark's, e.g. `{"motorbike": "motorcycle"}` |

Never use `eval_predict` or `EVAL` to raise a score. They exist to match the official evaluation.

**Its environment** is `models/<id>/pyproject.toml`.
- It depends on `mtr-measure` by path, and pins the official package and torch at exact versions.
- `.python-version` pins Python.
- The first `mtr check` writes `uv.lock`. Commit it.
- Every later command runs from the committed lock (`uv run --frozen`). `mtr check` refuses a lock that no longer matches `pyproject.toml` (`LOCK_STALE`).

## Language models

A language model has no code. The pinned llama.cpp build runs its GGUF files with the protocol's settings.

```python
KIND = "llm"
TASK = "chat"
SIZES = ["8b-q4_k_m"]            # one row per GGUF file: <parameters>-<quant>, lowercase
WEIGHTS = {
    "8b-q4_k_m": {
        "repo": "Qwen/Qwen3-8B-GGUF",
        "revision": "<40-hex commit>",   # "sha" in https://huggingface.co/api/models/<repo>
        "files": ["Qwen3-8B-Q4_K_M.gguf"],   # main file first; split files in order
        "quant": "Q4_K_M",
    },
}
SERVER_ARGS: list[str] = []        # only llama-server flags the model card requires, e.g. ["--jinja"]
THINKING_OFF_ARGS: list[str] = []  # how the model card turns thinking off, e.g. ["--chat-template-kwargs", '{"enable_thinking": false}']
```

- **Whose GGUF.** Prefer the one the model's own organization publishes. If there is none, a third-party quantization may be used; say so in the pull request.
- **Protocol flags are refused.** Flags the protocol owns, such as context, GPU layers and batch sizes, are refused in `SERVER_ARGS`.

## Prediction types

`to_prediction()` returns one of the standard types in `mtr_measure.predictions`.
- `mtr eval` and `mtr outputs` read predictions through it.
- `mtr check` checks that it returns the task's type.
- Timing never calls it.

| Task | Type | Fields |
|---|---|---|
| `object-detection` | `Boxes` | `xyxy` (N, 4), `scores` (N), `labels` |
| `instance-segmentation`, `semantic-segmentation` | `Masks` | `masks` (N, H, W) booleans, `labels`, `scores` or none. `Masks.from_class_map()` builds one mask per class from a class map |
| `image-matting` | `Alpha` | `alpha` (H, W) in [0, 1] |
| `depth-estimation` | `Depth` | `depth` (H, W), `inverse` for disparity |
| `pose-estimation` | `Keypoints` | `points` (N, K, 2), `scores` (N), `names`, `skeleton`, `visible` (N, K) or none |
| `ocr` | `Text` | `polygons` (N, P, 2), `texts`, `scores` (N) |

The rules that keep models comparable:
- **Pixels of the input image.** Coordinates and maps are in pixels of the input image as its file decodes. If the model works at another resolution, `to_prediction()` maps back. The drawing code checks the size and never resizes.
- **Class names, not ids.** A class then gets the same color in every model, and benchmarks map it by name.
- **Every score.**
  - The tool applies the task's fixed threshold (`docs/protocol.md`, "Fixed thresholds") and keeps scores at or above it.
  - If `predict()` itself filters by score, its cut-off must not be above the task's threshold.
- **Scores are probabilities** in [0, 1], not logits.
- **Visible keypoints.** Set `Keypoints.visible` only when the official code hides some keypoints when it draws them. For example, Ultralytics hides keypoints below a confidence of 0.25.

## Drawing

`mtr outputs` draws every model of a task the same way (`tools/measure/src/mtr_measure/render.py`), at the input image's exact size, as WebP.

| Type | How it is drawn |
|---|---|
| Boxes | An outline in the class color, and a tag with the class name and score |
| Masks | The photo dimmed, each mask tinted and outlined in its class color, with a tag on its topmost pixel |
| Alpha | The matte itself, white on black, so soft edges stay visible |
| Depth | The turbo colormap, near in red and far in dark blue. Each map is scaled between its own 2nd and 98th percentiles of nearness |
| Keypoints | The photo dimmed, a pink skeleton with white keypoints, and the person's score |

`Text` has no drawing yet.

## Working examples

- **`tools/measure/examples/`.** torchvision's ResNet-50, Faster R-CNN and DeepLabV3. They are small and need nothing but torchvision.
- **`models/*/`.** Every catalog model:
  - an installed package: `rf-detr`, `yolo26`
  - a pinned git checkout: `gtr`, `deimv2`
  - transformers: `grounding-dino`, `eomt`
  - a GGUF language model: `qwen3`
