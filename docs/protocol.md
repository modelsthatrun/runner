# Measurement protocol

Numbers compare only when every run follows the same protocol. `mtr` enforces this one and records everything it measured. The constants live in one file, `tools/measure/src/mtr_measure/protocol.py`, and every record carries its `PROTOCOL_VERSION`. Records without a `protocol` field were taken under version 1, which ran once instead of three times.

`mtr` measures three things:
- peak VRAM
- speed
- accuracy on a standard benchmark

It handles two kinds of model:
- **Image models** (`KIND = "image"`, the default). They run through a Python adapter, each in its own uv environment from a committed `uv.lock` (`docs/adapters.md`).
- **Language models** (`KIND = "llm"`). They run as GGUF files through one pinned llama.cpp build.

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
- **Precision.** fp16 if the official code supports it, otherwise fp32. The precision used is always recorded.
  - The adapter lists what the official code supports in `PRECISIONS`, and the tool picks the precision.
  - If fp16 fails, the run fails. It never falls back to fp32 on its own.

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

## Accuracy

`mtr eval <model> --size <size> --benchmark <benchmark>` scores a checkpoint on a standard public benchmark. It writes `models/<id>/evals/<size>/<benchmark>.json`, an eval record.

- **Same setup as speed.** It uses the same adapter, environment and precision or quantization as the timed run. It records every setting that affects the score.
- **Nothing is paused.** Nothing is timed. It first checks that the GPU has room next to the other programs: what `check` saw, plus 1 GiB.
- **Image models.**
  - The benchmarks are `coco-val2017-bbox` and `coco-val2017-segm`, scored with pycocotools, the reference implementation.
  - The tool reads predictions through `to_prediction()`. It never applies the display threshold, and it keeps the 100 highest-scoring detections per image.
  - If the official evaluation calls the model differently from its README, the adapter's `eval_predict()` does it that way. Class names map to COCO by name, with the adapter's `EVAL["label_map"]` for other names.
- **Language models.**
  - lm-evaluation-harness 0.4.13 (`tools/measure/lmeval/`) runs against the same llama.cpp build, through its chat endpoint, with the model's own chat template.
  - The suite is IFEval and GSM8K (`gsm8k_cot_llama`, 8 worked examples as a chat), with thinking off. It takes about 11 minutes per 8B checkpoint.
- **Data and evidence.**
  - Benchmark data downloads once into `~/.cache/modelsthatrun/datasets/` and is never published.
  - Raw predictions stay next to the record as evidence. The record keeps their sha256.

## Outputs

`mtr outputs` draws a checkpoint's outputs on a task's test images, for the catalog's galleries. The catalog's maintainers run it: the shared test set isn't part of this repository, and submissions don't include outputs.

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
   - It loads the model, makes a few calls, validates the output types and shows approximate numbers.
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
6. **The result line.** Every command ends with one `mtr-result: {json}` line: status, code, message, hint, outputs and warnings (`errors.py`). The exit code is fixed per code.
7. **Freeing disk.** `mtr clean <model> --yes` frees what a model used.
