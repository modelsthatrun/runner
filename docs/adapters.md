# Adapters

Every model family has one adapter, `models/<id>/adapter.py`. It tells the shared tool how to load a checkpoint and how to make one call from input to output, the way the model's official README does. The tool knows nothing else about the model.
- `mtr check`, `mtr run`, `mtr eval` and `mtr profile` use the adapter.
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

**Precision.** The tool picks fp16 if it's listed, otherwise bf16, otherwise fp32. If the official inference runs bf16, list `bf16` and leave out `fp16`, even if fp16 could be forced. numpy can't read a bf16 tensor, so `to_prediction()` calls `.float()` before `.numpy()`.

**Optional:**

| Name | Use it when |
|---|---|
| `prepare_input(paths)` | The model takes several files, or an input that isn't one RGB image. By default `x` is the input file decoded as an RGB PIL image |
| `input_size(size, x)` | The resolution depends on the input. Use it instead of `INPUT` |
| `package_version()` | The model runs from a git checkout rather than an installed package. Use it instead of `PACKAGE`, e.g. `"owner/repo@<commit>"` |
| `LATENCY_SCOPE` | One `predict()` call covers something other than pre-processing, forward and post-processing |
| `eval_predict(model, x)` | The official evaluation calls the model differently from its README, e.g. a lower score threshold. `mtr eval` uses it instead of `predict()` |
| `EVAL = {"label_map": {...}}` | The model's class names differ from the benchmark's, e.g. `{"motorbike": "motorcycle"}` |
| `preprocess(model, x)`, `forward(model, inp)`, `postprocess(model, raw, x)` | You can split `predict()` into its phases, for `mtr profile`'s phase split ([protocol](protocol.md), "Profiles"). `predict(model, x)` must return `postprocess(model, forward(model, preprocess(model, x)), x)`; the simplest way is to define it as exactly that. The split needs all three. With fewer there is no split, and the functions are the adapter's own helpers |
| `phase_times(model, out)` | Instead of the three above, when the official package times its own phases and closes each with a synchronize. Return `{"preprocess": ms, "forward": ms, "postprocess": ms}` for the call that returned `out`. `models/yolo26/adapter.py` reads Ultralytics' `Results.speed` |

Never use `eval_predict` or `EVAL` to raise a score. They exist to match the official evaluation.

**Its environment** is `models/<id>/pyproject.toml`.
- It depends on `mtr-measure` by path, and pins the official package and torch at exact versions.
- `.python-version` pins Python.
- The first `mtr check` writes `uv.lock`. Commit it.
- Every later command runs from the committed lock (`uv run --frozen`). `mtr check` refuses a lock that no longer matches `pyproject.toml` (`LOCK_STALE`).

### A TensorRT row

Add one only where the authors publish the whole path: an export to ONNX, a TensorRT build step, and code that runs the engine ([protocol](protocol.md), "Runtimes"). `models/d-fine/adapter.py` is the example of a `trtexec` command, `models/yolo26/adapter.py` of a package that builds the engine itself.

```python
TENSORRT = {"precision": "fp16", "builder": {"fp16": True}, "status": "runs-as-is"}

def export_onnx(size: str, out: pathlib.Path) -> dict: ...    # the official export on the released checkpoint, into `out`
# or, when one official call exports and builds:
def export_engine(size: str, out: pathlib.Path) -> dict: ...  # that call, on the released checkpoint, into `out`
def load_engine(size: str, engine: pathlib.Path) -> Model: ... # the engine in the official TensorRT inference code
```

- **`TENSORRT`.** `precision` is what the official TensorRT step builds, and `builder` holds only the flags its command names (`fp16`, `bf16`, `int8`). `status` is `runs-with-fixes`, with `"patch": "fixes/<file>"`, when the official path needed a fix. Add `"input": "640x640"` (or one per size) when the engine's input, fixed at export, differs from the PyTorch row's: the run records it.
- **`export_onnx()`** runs the authors' export script or API with the arguments their README gives, on the released checkpoint, and returns `{"onnx": path, "exporter": "owner/repo@<commit> tools/export.py", "args": [...]}`. Record file names in `args`, not paths on your machine. The tool builds the engine from the returned file.
- **`export_engine()`** is for a package whose official TensorRT step is one call that exports and builds, such as Ultralytics' `export(format="engine")`. Define it instead of `export_onnx()`. It returns what `export_onnx()` returns plus `"engine": path`, and optionally `"builder": {...}` with the settings the call passes TensorRT. The record says the package built the engine, with its own settings.
- **`load_engine()`** returns a model that `predict()`, `eval_predict()` and `to_prediction()` take as they take `load()`'s. Usually they check which one they got and call the official TensorRT inference code for the engine.
- **Its environment.** Put TensorRT and whatever the export and inference code import in a dependency group named `tensorrt`, and take NVIDIA's packages from its index:

  ```toml
  [dependency-groups]
  tensorrt = ["tensorrt-cu13==10.16.1.11", "tensorrt-cu13-libs==10.16.1.11", "tensorrt-cu13-bindings==10.16.1.11", "onnx"]

  [[tool.uv.index]]
  name = "nvidia"
  url = "https://pypi.nvidia.com"
  explicit = true

  [tool.uv.sources]
  tensorrt-cu13-libs = { index = "nvidia" }
  tensorrt-cu13-bindings = { index = "nvidia" }
  ```

  Use `tensorrt-cu12` for a torch built for CUDA 12. After `uv lock`, check that every package the PyTorch row uses kept its version.
- **Order.** `mtr check --runtime tensorrt`, then `mtr eval --runtime tensorrt`, then the timed run. The queue refuses an engine that wasn't scored.

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

## Text-to-image models

A text-to-image model runs its pipeline through an adapter like an image model's, with `KIND = "generation"`. Start from `models/_template-gen/` (`mtr new <id> --kind generation`).

```python
KIND = "generation"
TASK = "text-to-image"
SIZES = ["4b"]
PRECISIONS = ["bf16"]             # what the model card loads it in
PACKAGE = "diffusers"
GENERATION = {"4b": {"resolution": "1024x1024", "steps": 4, "guidance": 1.0, "generator": "cuda"}}

def load(size, precision):        # the pipeline, every component on the GPU
    return Flux2KleinPipeline.from_pretrained(REPO, revision=REVISION, torch_dtype=torch.bfloat16).to("cuda")

@torch.inference_mode()
def predict(model, x):            # x: mtr_measure.generation.Generate
    return model(prompt=x.prompt, width=x.width, height=x.height, num_inference_steps=x.steps, guidance_scale=x.guidance,
                 generator=torch.Generator(x.generator).manual_seed(x.seed)).images[0]

def weight_files(size): ...       # every weight file load() read, the denoiser's first
```

- **Settings from the model card.** `GENERATION` holds its example's resolution, steps and guidance, and the device it seeds its generator on. Where the card relies on the pipeline's defaults, write those down.
- **The tool makes the input.** `predict()` gets the prompt and the settings in `x`, and must use them as they are. It returns one PIL image at `x.width` × `x.height`; `mtr check` refuses anything else. `mtr eval --benchmark geneval` calls the same `predict()` with GenEval's prompts and seeds 42 to 45, so the generator must be seeded from `x.seed`.
- **No offload.** Leave out `enable_model_cpu_offload()`, `enable_sequential_cpu_offload()` and the like, even when the card calls them. Everything stays on the GPU.
- **The scheduler.** The record names the class of the pipeline's `scheduler`. A pipeline without one sets `SCHEDULER` to the sampler's name.
- **Pin the revision.** Load from the repository's 40-hex commit, and list each weight file at that commit in `weight_files()`.
- **No `to_prediction()`.** The image is the output.

## Prediction types

`to_prediction()` returns one of the standard types in `mtr_measure.predictions`.
- `mtr eval` and `mtr outputs` read predictions through it.
- `mtr check` checks that it returns the task's type, and that `predict()` finds something on the task's timing input.
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
