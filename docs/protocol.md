# Measurement protocol

Numbers compare only when every run follows the same protocol. `mtr` enforces this one and records everything it measured. The constants live in one file, `tools/measure/src/mtr_measure/protocol.py`, and every record carries its `PROTOCOL_VERSION`. Records without a `protocol` field were taken under version 1, which ran once instead of three times.

`mtr` measures three things:
- peak VRAM
- speed
- accuracy on a standard benchmark

It handles three kinds of model:
- **Image models** (`KIND = "image"`, the default). They run through a Python adapter, each in its own uv environment from a committed `uv.lock` (`docs/adapters.md`).
- **Language models** (`KIND = "llm"`). They run as GGUF files through one pinned llama.cpp build.
- **Text-to-image models** (`KIND = "generation"`). They run a pipeline, such as a diffusers one, through a Python adapter like an image model's, and make one image per call ("Image generation" below).

An image model can also have a TensorRT row: the same checkpoint, run as an engine built from the authors' own ONNX export ("Runtimes" below).

## The machine

- **What it needs.** Linux and an NVIDIA GPU with its driver. `mtr doctor` checks the machine and shows what a record will say about it.
- **What each run records:**
  - the GPU name, driver, total memory, compute capability and power limit
  - the CPU model and the OS
  - the idle baseline
- **Other programs on the GPU.** A desktop on the measuring GPU holds some memory: Xorg, a compositor, a browser. That is fine, because VRAM is measured over the idle baseline. Things that make a timed run refuse to start:
  - another program **computes** on the GPU
  - the idle GPU is more than 10% busy
  - idle memory moves by more than 64 MiB within a second
  
  The tool never closes programs. It says which one is in the way.
- **The CPU check.** Timed calls include CPU work. So before each run, and across each run, the tool measures what share of the CPU other programs used (`cpu.py`), leaving out its own share and the worker's. Above 15% of all logical CPUs, the run fails with `CPU_BUSY`. An idle desktop uses 1–5%.
- **Pause hooks.** A machine that streams its desktop or runs a game launcher on the measuring GPU can name pause hooks in `~/.config/modelsthatrun/machine.toml`, for example `pause = ["sunshine", "steam"]`.
  - The hooks stop those programs for each timed run and start them again as soon as it ends, even if it fails.
  - The Sunshine hook also arms a fallback timer that starts Sunshine again after 10 minutes, in case the tool dies first.
  - **Sunshine:** a systemd user unit, either the Flatpak's or the native packages' `sunshine.service`.
  - **Steam:** the Flatpak or the native client.
  - **Other programs** have no hook. Close them before a timed run.
  - Without the file, nothing is paused.
- **The reference machine.** It is an RTX 4090 (24 GB) that also drives a desktop. Its idle baseline with the hooks paused is about 800 MiB. Numbers on the site under "Measured · our RTX 4090" come from it. Community results come from contributors' machines and are labelled with their GPU.

## Scope

- Inference only, batch size 1. Image models use the official default input resolution.
- **Precision.** fp16 if the official code supports it, otherwise bf16 if it does, otherwise fp32. The precision used is always recorded.
  - The adapter lists what the official code supports in `PRECISIONS`, and the tool picks the precision.
  - If the official inference runs bf16 on the GPU, the adapter lists `bf16` and leaves out `fp16`, even if fp16 could be forced.
  - If the chosen precision fails, the run fails. It never falls back to another on its own.
  - bf16 has to be native. On a GPU that only emulates it, the run fails.

## Latency

- **What is timed.** One call to the adapter's `predict()`. That covers pre-processing, the forward pass and post-processing, which is what a user's code actually experiences. The record says so in `latency_scope`.
- **Input decoding is not timed.** The input file is decoded before timing starts, because decoding isn't part of the model.
- **The timer.** `torch.cuda.synchronize()` is called before starting and before stopping the timer, and the clock is `time.perf_counter()`.
- **One fixed input.** Every model of a task is timed on that task's timing image, `timing/timing.toml`. The tool checks the file's sha256 before it times anything.
- **Three independent runs.** The tool starts the worker process three times.
  - Each run loads the model, then makes 10 warm-up calls and 100 timed calls. Its median and p90 are in milliseconds. The p90 interpolates linearly between ranks, as numpy does by default.
  - The record's median and p90 are the medians of the three runs' values, and `runs` keeps each run's numbers.
  - Separate processes catch what changes between processes: kernel selection, memory layout and clock state.
  - `runs/<size>.times.json` keeps every timed call of every run, so anyone can recompute the record's numbers.
- **The spread limit.** If the three medians spread by more than `MAX_SPREAD` ((max − min) / median), no record is written (`VARIANCE`). The limit is 10%. Honest spreads are 0.2–6.8% on the reference machine, so speed differences under 10% between models may be noise.

## VRAM

- **Polling.** NVML reports current usage, not the peak. So the tool polls device-level used memory every 5 ms (the protocol allows 10 ms or less), from just before the model loads until it is done.
- **Separate processes.** The model runs in a worker process, and the polling runs in the parent. Polling therefore never competes with the timed loop.
- **Device level.** Memory is measured at the device level, because per-process entries are often missing: `vram_peak_mib = max(device used) - baseline`.
  - The baseline is the median of one second of samples taken before the worker starts.
  - The peak includes the worker's CUDA context, which a user's process has too. For a small model the context can be most of the peak.
  - Each run's peak is taken over its own baseline, and the record keeps the highest.
  - The same method measures the llama-server process that runs a language model.
- **Drift.** After the worker exits, the tool takes the baseline again. If it moved by more than 64 MiB, something else used the GPU during the run, and no record is written.
- **Cross-check.** `torch.cuda.max_memory_reserved()` is recorded next to the peak.
- **Polling gaps.** A run fails if the 99th-percentile polling gap exceeds 10 ms.

## Environment and provenance

Every record carries:
- **The environment.**
  - the GPU name and driver version, total GPU memory, compute capability and power limits, from NVML
  - the CPU model, and the OS with its C library
  - the CUDA runtime version that torch was built with (`torch.version.cuda`), not the "CUDA Version" that nvidia-smi prints
  - the torch, Python and model package versions
  - each weight file's sha256 and size
  - the timestamp
- **Its provenance:**
  - the tool's git commit, and whether `tools/measure/`, `timing/` or the model directory had uncommitted changes
  - the sha256 of `adapter.py` and `uv.lock`
  - the sha256 of each input file

Records never carry a host name, a user name or a path.

## Language models

- **Engine.** One pinned llama.cpp build runs every language model: release v0.7.0, build b11669 (`engine.py`).
  - The official prebuilt Linux binaries need glibc 2.38, so the tool builds the pinned source once with the local CUDA toolkit (`nvcc`), for the local GPU's architecture only.
  - The build goes into `~/.cache/modelsthatrun/engines/` and takes a few minutes.
  - Records carry the build and the binary's sha256.
- **Weights.** One GGUF file is one checkpoint row. The adapter pins its repository, its commit and the file names. The tool downloads them into `~/.cache/modelsthatrun/weights/<id>/` and checks each file against the sha256 that Hugging Face reports.
- **Server.** `llama-server -c 8192 -np 1 -ngl 999 --fit off --no-webui -lv 4`:
  - one slot with 8192 tokens of context
  - every layer on the GPU, and llama.cpp may not shrink anything to fit
  - Batch and cache settings stay at the build's defaults.
  - An adapter may add only flags that the model card requires. The flags the protocol owns are refused.
  - If any layer doesn't fit on the GPU, the result is `OOM`, and no record is written.
- **Workload.**
  - The prompt is the first 512 tokens of `tools/measure/src/mtr_measure/llm_prompt.txt`, as the model's tokenizer splits it, sent as token ids.
  - It generates 256 tokens, greedy, with the end-of-sequence token ignored.
  - Prompt caching is off, so every request runs the whole prefill.
  - Each run makes 2 warm-up and 10 timed requests and starts a fresh server, with three independent runs as for image models.
- **Numbers.**
  - `ttft_ms`: time to the first streamed token, client side, median and p90
  - `prefill_tps` and `decode_tps`: the server's own timings, medians
  - `e2e_ms`: the whole request
  - `vram_peak_mib`: at 8192 tokens of context
  - The spread limit applies to the decode rate.
- **Prompt set.** `tools/measure/src/mtr_measure/prompts/chat.toml` holds the chat task's shared prompts, each with the check a correct answer must pass.
  - Every checkpoint answers each one greedily, with thinking off, no system prompt and at most 512 tokens.
  - Changing the set raises its `version`, and every chat checkpoint answers again.

## Image generation

Text-to-image models are timed like image models (Latency, VRAM), with these differences:

- **One call is one image.** It covers the text encoding, every denoising step and the VAE decode, up to the PIL image the pipeline returns. The prompt is read before timing starts. Saving the image isn't timed.
- **What the tool passes.** The tool, not the adapter, makes each call's input: the prompt, the seed, and the steps, guidance, width and height the adapter declares in `GENERATION`. The record's `generation` block repeats them with the device the generator is seeded on and the pipeline's scheduler. `input` is the image's resolution.
- **The official settings.** Resolution, steps and guidance come from the official model card's example. Where the card leaves them out, the pipeline's own defaults apply, and the adapter writes them down.
- **Everything on the GPU.** Every component stays on the GPU for the whole call. Calls such as `enable_model_cpu_offload()`, which model cards add to save memory, are left out. A model that doesn't fit without them doesn't run on this GPU (`OOM`).
- **One fixed prompt.** Every model is timed on `timing/bicycle-street.txt`, with seed 0. The tool checks its sha256 first.
- **Fewer calls.** A call takes seconds, so each run makes 2 warm-up and 10 timed calls, as a language model's run makes requests. There are still three independent runs, and the spread limit applies to their medians.
- **The prompt set.** `tools/measure/src/mtr_measure/prompts/text-to-image.toml` holds the task's shared prompts, each with the check a correct image must pass. `mtr outputs` makes one image per prompt with the set's seed and the same settings as a timed call.
  - Changing the set raises its `version`, and every text-to-image checkpoint makes its images again.
  - The same seed doesn't give the same pixels on another GPU, driver or library version, or even in another process. Images are compared by eye, never by hash.
- **Accuracy.** GenEval (Accuracy, below).
- **Profiles** make fewer calls (Profiles, below).

## Runtimes

A runtime is how a checkpoint's call executes. Every record names one, or means the default.

- **`pytorch-cuda`, the default.** The official PyTorch code, eager, on the GPU. A record without a `runtime` block is this one.
- **`tensorrt`.** An engine built from the authors' own ONNX export, with the pinned TensorRT ("The environment", below), run the way their TensorRT inference code runs it. It is another measurement of the same checkpoint: the weights, the license and the outputs stay the checkpoint's.
- **Other kinds keep their runtime.** Language models already run on llama.cpp, and text-to-image models on their PyTorch pipeline.

**The official path only.** A model has a TensorRT row only where its authors publish the whole path: an export script or export API, and code that runs the result. The adapter calls them with the arguments their README gives (`docs/adapters.md`).
- Nothing of our own: no export of ours, no edits to the ONNX graph, no plugins, and no builder settings the docs don't name.
- The export always loads the released checkpoint the row is for, even where a README's example leaves the weights out.
- Where the official path needs a fix, the row is `runs-with-fixes`, and the adapter and the record say so.
- Pre- and post-processing are the official TensorRT inference code's. What a call covers is unchanged: `predict()`, timed with the same synchronize, which also waits for the stream TensorRT runs on.
- **Precision** is what the official TensorRT step asks for: `--fp16`, `half=True` and the like. TensorRT may still keep some layers in fp32. The record states the builder flag, not each layer's type.
- **Input.** An engine's input is fixed at export, and the record states the input it ran at.
- **INT8** is left out. It needs calibration images.

**Building the engine.** The TensorRT wheels hold no `trtexec`. Where a README says `trtexec --onnx=… --saveEngine=… --fp16`, the tool builds with TensorRT's Python API and sets it as trtexec sets it (NVIDIA's `samples/common/sampleEngines.cpp`): the flags the command names, trtexec's 8 timing iterations per tactic, and, for an input whose ONNX shape is dynamic, one optimization profile with every dynamic dimension at 1. Everything else stays at TensorRT's defaults. The record lists these settings.
- Where the official step is one package call that exports and builds, such as Ultralytics' `export(format="engine")`, that call builds the engine, with the package's own settings, and the record says so.
- `mtr check --runtime tensorrt` exports and builds, in its own process before the check's calls, so the builder's memory never shows up in the check's numbers. It pauses nothing, and refuses to build while another program computes on the GPU or the idle GPU is more than 10% busy, because TensorRT picks kernels by timing them.
- It keeps the engine, and builds it again only with `--rebuild`, or when the weights, the export, the builder settings, the TensorRT version or the GPU changed.
- Engines and ONNX files stay in `~/.cache/modelsthatrun/trt/<model>/<size>/`. They are never published or submitted. `mtr clean` removes them.
- The ONNX file must hold its own weights, so its sha256 covers them.

**What the record says.** A TensorRT record lists the checkpoint's own weight files, as every record does, and adds a `runtime` block: the TensorRT version, the exporter with its arguments, the opset the file declares, the ONNX file's sha256 and size, and the engine's sha256, size, builder settings, build time and date.
- TensorRT picks kernels by timing them, so a rebuild gives other bytes, on another GPU and even on the same one. The engine's hash ties a run record to its eval on one machine. The ONNX hash, with the export settings, is what another machine can reproduce.
- VRAM is measured as for every record, at the device level over the idle baseline. `torch_max_reserved_mib` is null, because TensorRT allocates outside PyTorch's allocator.
- The three runs load the same engine in three fresh processes, so their spread covers loading and running, not the build's choice of kernels.

**Scored before it is timed.** A new runtime is scored on the released weights before any timed run of it. A check can pass on a model whose outputs are all NaN: a score threshold drops NaN scores, and finding nothing on one image is a valid prediction. Only an eval, which scores every candidate on thousands of images, shows it.
- `mtr eval --runtime tensorrt` scores the engine the check built and names its sha256 in the eval record.
- `mtr queue add --runtime tensorrt` and `mtr run --runtime tensorrt` refuse a checkpoint until an eval record names the same engine (`NOT_SCORED`). `mtr queue run` checks again before timing, in case the engine was built again after it was queued.
- `mtr check --runtime tensorrt` also fails when the engine's `predict()` finds nothing on the task's timing input, as for any check. That catches outputs that are empty. The eval catches outputs that are wrong but not empty.
- An engine's score is its own. It never borrows the PyTorch row's, because the precision differs and an fp16 engine can be wrong.

**The environment.** TensorRT sits in an optional dependency group, `tensorrt`, in the model's own `pyproject.toml`, with the libraries from NVIDIA's index (`https://pypi.nvidia.com`) pinned in `uv.lock` with their hashes. `--runtime tensorrt` adds `--group tensorrt` to `uv run`.
- Adding the group must leave every other locked package at its version.
- **One version, with exceptions.** Every model pins TensorRT 10.16.1.11, unless an eval shows that its engines give wrong outputs for that model while an earlier release gives the right ones. That model then pins the newest release whose engines score like its PyTorch row, says why in its `pyproject.toml`, and its records name the version. Rows of one runtime may therefore differ in TensorRT version, as PyTorch rows differ in torch version.
  - D-FINE pins 10.13.2.6. On 10.16.1.11 its fp16 engines lose 5–13 COCO AP for S, M, L and X, and N's is right. On S, every release tried from 10.13.3.9 to 10.16.1.11 loses about 9 AP even in FP32, NVIDIA's own `trtexec` included, while 10.13.2.6, 10.4 (the version its README uses) and ONNX Runtime on the same file match PyTorch. On 10.13.2.6 every size's fp16 engine scores within 0.1 AP of PyTorch. 10.13.2.6's library carries kernels for sm_75 to sm_120, as 10.16.1.11's does.
- `uv run` doesn't remove packages a command doesn't ask for, so after a TensorRT command the group stays installed in the model's environment. PyTorch rows never import it.
- **License.** TensorRT is NVIDIA's proprietary SDK. Installing the group means accepting the NVIDIA Software License Agreement and its TensorRT supplement, in `LICENSE.txt` of the `tensorrt` package. Among other terms it allows use only on systems with NVIDIA GPUs and lets the software collect non-personal information about the system. Read it before you install.

**Community submissions** of TensorRT records aren't open yet. `mtr submit` sends only `runs/<size>.json`.

## Accuracy

`mtr eval <model> --size <size> --benchmark <benchmark>` scores a checkpoint on a standard public benchmark. It writes `models/<id>/evals/<size>/<benchmark>.json`, an eval record.

- **Same setup as speed.** It uses the same adapter, environment and precision or quantization as the timed run. It records every setting that affects the score.
- **Nothing is paused.** Nothing is timed. It first checks that the GPU has room next to the other programs: what `check` saw, plus 1 GiB.
- **Image models.**
  - The benchmarks are `coco-val2017-bbox` and `coco-val2017-segm`, scored with pycocotools, the reference implementation.
  - The tool reads predictions through `to_prediction()`. It never applies the display threshold, and it keeps the 100 highest-scoring detections per image.
  - If the official evaluation calls the model differently from its README, the adapter's `eval_predict()` does it that way. Class names map to COCO by name, with the adapter's `EVAL["label_map"]` for other names.
  - With `--runtime tensorrt` it scores the engine `check` built, at the engine's precision, and writes `evals/<size>/<benchmark>.tensorrt.json` (Runtimes, above).
- **Language models.**
  - lm-evaluation-harness 0.4.13 (`tools/measure/lmeval/`) runs against the same llama.cpp build, through its chat endpoint, with the model's own chat template.
  - The suite is IFEval and GSM8K (`gsm8k_cot_llama`, 8 worked examples as a chat), with thinking off. It takes about 11 minutes per 8B checkpoint.
  - An answer may run to 1280 tokens on both. IFEval's own configuration sets that limit. GSM8K's sets none, and lm-eval's default of 256 tokens cuts off models that reason at length, so the tool gives it the same limit. Each eval record states it in `settings.generation`.
- **Text-to-image models.**
  - The benchmark is GenEval (https://github.com/djghosh13/geneval, MIT): 553 prompts that ask for objects, counts, colors and positions, four images each. The score is the mean of its six tasks' shares of correct images, 0–1 as GenEval reports it, and the record keeps each task's share.
  - The images are made like a timed call: the prompt as GenEval wrote it, the adapter's `GENERATION` settings, no prompt rewriter and no negative prompt. The four images of every prompt are seeded 42, 43, 44 and 45, each with its own generator. GenEval's own script seeds 42 once per prompt and draws the four in a row, so the structure is the same and the pixels aren't.
  - They are scored by GenEval's own `evaluate_images.py`, unchanged and at a pinned commit, with its default options: Mask2Former (Swin-S, MMDetection) finds the objects and CLIP ViT-L/14 names their colors. It runs in its own environment, `tools/measure/geneval/`, with the versions GenEval's `environment.yml` pins: Python 3.9, torch 1.12.1 with CUDA 11.3, mmcv-full 1.7.1, and mmdetection 2.28.2, the last release of the `2.x` branch its README installs. One library is newer: torch 1.12 compiles a few kernels when first used (Mask2Former's integer product of feature-map shapes, in a trial over all six tasks), and the NVRTC 11.3 it bundles doesn't know RTX 40 GPUs (sm_89). The tool preloads NVIDIA's NVRTC 11.8 for the evaluator, so the same kernel source is compiled by a newer compiler. torch 1.12 has GPU code up to sm_86, so GenEval runs on RTX 30 and RTX 40 GPUs, and not on RTX 50. The tool also runs GenEval's `summary_scores.py` and checks that it gives the same overall score.
  - The images are PNG files next to the record, made once: a run that stops resumes, and images made with other settings are refused. On an RTX 4090 a four-step model takes about an hour and a 50-step one about five.
  - An eval pauses nothing, so before the images it wants the model's memory (what `check` saw) plus 1 GB free, room for the programs that keep running. A model that only fits with the pause hooks stopped, as for its timed runs, can be scored with `--pause`. The hooks then hold for the whole image step, which takes hours, and the margin is 512 MiB, room for the desktop. Scoring runs after they start again. The margin decides only whether the run starts, never a score.
- **Data and evidence.**
  - Benchmark data downloads once into `~/.cache/modelsthatrun/datasets/` and is never published.
  - Raw predictions stay next to the record as evidence: detections, lm-eval's samples, or GenEval's images and results. The record keeps their sha256.

## Profiles

`mtr profile <model> --size <size>` splits an image model's GPU time by kernel. It writes `models/<id>/profiles/<size>.json`, a profile record. ModelsThatRun uses it to estimate speed on other GPUs: each kernel's time is scaled by the unit it ran on, with the other GPU's spec sheet, and the CPU's part of a call is held as it is.

- **Not a timed run.** Nothing is paused. It stops if the idle GPU is more than 10% busy, and warns about other processes that hold the GPU.
- **Same setup as speed.** It uses the same adapter, environment, precision and input as `mtr run`.
- **The calls.**
  1. Load the checkpoint, then 10 warm-up calls.
  2. 50 plain calls, timed as in a timed run, without the profiler.
  3. If the adapter splits a call into phases, 20 phased calls (see below).
  4. 23 calls under `torch.profiler`. The first 3 are dropped.

  A text-to-image call takes seconds and launches hundreds of thousands of kernels, so it makes 2 warm-up, 5 plain and 1 + 3 profiled calls.
- **Each profiled call is one window.** It is closed by a synchronize, so every kernel and copy it started ran inside it. Per kernel name, the record keeps the median over the windows of:
  - its launches
  - its GPU time
  - its GPU-bound time: the time of launches that the GPU went straight on from, because the next work was already queued (started within 2 µs) or the call was waiting for it
- **Phases.** A phased call is split into pre-processing (from the decoded input to the tensor on the GPU), the network, and post-processing (from the network's output to `predict()`'s result). The record keeps each phase's median, and the median of the whole call.
  - Who splits it:
    - the adapter's `preprocess()`, `forward()` and `postprocess()`, with a synchronize after each phase (`source: "adapter"`)
    - or the official package's own phase timer, read with `phase_times()`, when it closes each phase with a synchronize itself (`source: "package"`; Ultralytics does)
  - The synchronizes stop the CPU and the GPU from working at the same time, so the phases needn't add up to a plain call. They show where a call's time goes, not a second latency.
  - A package's timer may leave some of the call outside its phases, such as reading the input. The whole call's median shows how much.
  - Models whose adapter doesn't split a call have no phases.
- **What it records about the build.** It records the PyTorch build's GPU architectures and the settings that decide whether fp32 work runs on tensor cores. Those settings are TF32 for matmuls and for cuDNN, and the fp32 matmul precision. It reads them after the model has loaded.
- **Its limits.**
  - Kernel times come from the GPU's own clock, so the profiler's overhead doesn't change them.
  - The overhead does slow the CPU, so fewer launches look GPU-bound than without it.
  - Language models need no profile.

## Outputs

`mtr outputs` draws a checkpoint's outputs on a task's test images, for the catalog's galleries. The catalog's maintainers run it: the shared test set isn't part of this repository, and submissions don't include outputs. Language models answer their prompt set instead (`answers.json`), and text-to-image models make one image per prompt of theirs (`<prompt-id>.webp` and `generations.json`).

- **Same treatment for every model.** Every model of a task is drawn the same way, with one fixed score threshold per task. Nobody tunes it per model.
- **Format.** Outputs are WebP, at the input image's own size.

### Fixed thresholds

| Task | Threshold | Applies to |
|---|---|---|
| object-detection | 0.5 | box score |
| instance-segmentation | 0.5 | mask score |
| pose-estimation | 0.5 | person score; keypoints are drawn as each model's official code shows them |

Matting and depth outputs have no scores, so they have no threshold.

## Running a measurement

1. **`mtr doctor`.** Is the machine ready?
2. **`mtr check <model> --size <size>`.**
   - It pauses nothing and writes no record.
   - It downloads the weights, and builds llama.cpp on first use, so nothing slow happens during a timed run.
   - It loads the model, makes a few calls, validates the outputs and shows approximate numbers.
   - The output must be the task's prediction type, with no NaN. On the task's timing input `predict()` must also find something: a call that finds nothing doesn't run the model's whole path, and a score threshold hides outputs that are all NaN.
   - A pass writes `.check/<size>.json`, tied to the hashes of `adapter.py` and `uv.lock`.
3. **`mtr run <model> --size <size>`** (or `mtr bench` for check and run together). For each of the three runs it:
   1. checks the machine
   2. takes the baseline
   3. runs the worker
   4. takes the baseline again
   
   Then it checks the spread, hashes the weights, validates the record and writes:
   - `models/<id>/runs/<size>.json`
   - `<size>.times.json`
   - `<size>.log`, with every timing and check
4. **If a check fails,** nothing is written, and `runs/<size>.failed.log` says why.
   - `--dry-run` measures without writing.
   - `--force` runs past failed checks without pausing anything, and never writes a record.
5. **Batching.** `mtr queue add` collects timed runs, and `mtr queue run` measures them in one pause.
6. **A TensorRT row** goes `mtr check --runtime tensorrt`, then `mtr eval --runtime tensorrt`, then `mtr queue add --runtime tensorrt`. Its files carry the runtime: `.check/<size>.tensorrt.json`, `evals/<size>/<benchmark>.tensorrt.json`, `runs/<size>.tensorrt.json` and `<size>.tensorrt.times.json`.
7. **The result line.** Every command ends with one `mtr-result: {json}` line: status, code, message, hint, outputs and warnings (`errors.py`). The exit code is fixed per code.
8. **Freeing disk.** `mtr clean <model> --yes` frees what a model used, TensorRT engines included.
