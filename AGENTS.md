# AGENTS.md

Instructions for AI coding agents working in this repository: Claude Code, Codex, Cursor and others. People should start with [README.md](README.md).

`mtr` measures open models on the user's NVIDIA GPU by one protocol ([docs/protocol.md](docs/protocol.md)) and sends the run records to ModelsThatRun for review. You will usually do one of two things:
- **Measure an existing model** on the user's GPU, and maybe submit it.
- **Add an adapter** for a new model.

## Rules

1. **Never write or edit a record by hand.** Records come only from the tool:
   - `models/*/runs/*.json` and `*.times.json`
   - `models/*/evals/**`
   - `.check/*.json`
   
   If a record is wrong, measure again. A hand-edited record fails the inbox's checks, and it would be a false claim under the user's name.
2. **Never change the protocol to make something pass.** Leave `tools/measure/src/mtr_measure/protocol.py` alone: repetitions, `MAX_SPREAD`, context, prompt and generation lengths, benchmark settings, thresholds. Leave `timing/` and `prompts/` alone too. When the tool refuses, report its code and hint.
3. **Submit only when the user says so in this conversation.**
   - `mtr submit` publishes under the user's GitHub name.
   - Run `mtr submit <model> --dry-run` first and show the user what would be sent.
   - Then run `mtr submit <model> --yes` only after they agree.
   - Never run `mtr login` yourself, even when a result's hint says to. It needs the user's browser and signs in as them. On `AUTH`, ask the user to run `./mtr login`.
4. **Ask before a timed run on a machine someone is using.**
   - `mtr run`, `mtr bench` and `mtr queue run` hold the GPU and need a quiet CPU for a few minutes each. Pause hooks in `machine.toml` may stop the user's streaming or game launcher for that time.
   - `mtr check` and `mtr eval` pause nothing.
5. **Never close the user's programs.** That covers games, videos, browsers and training jobs. If the GPU or the CPU is busy (`GPU_BUSY`, `CPU_BUSY`, `VRAM_LOW`), say which program it is and let the user decide.
6. **Never lower settings to make a model fit.** `OOM` at the protocol's settings is a result: the checkpoint doesn't run on this GPU.
7. **Licenses come from primary sources only.** Take them from the repository's LICENSE file and from the model card of that exact checkpoint, with the URL and the date you read them. Never use blogs, vendor posts or memory.
8. **Keep the checkout clean.** Records from a modified `tools/measure/`, `timing/` or model directory can't be submitted (`DIRTY`). Don't change tool code or adapters while measuring someone else's model. If one needs a change, propose a pull request to the user. Never push or open one without asking.

## Reading results

Every command ends with exactly one line:

```
mtr-result: {"status": "ok"|"fail", "code": "...", "message": "...", "hint": "...", "outputs": {...}, "warnings": [...]}
```

- **How to read it.** Use `| tail -1` and parse the JSON. Don't scrape the log above it.
- **The exit code** is fixed per `code` (`tools/measure/src/mtr_measure/errors.py`). A pipe hides it, so read `code` from the JSON instead.
- **The hint** says what to do next.
- **Logs** sit next to the outputs:
  - `models/<id>/.check/<size>.log`
  - `runs/<size>.log`, or `runs/<size>.failed.log` when a run failed
  - `evals/<size>/<benchmark>.log`

## Measuring an existing model

```bash
./mtr doctor | tail -1                          # outputs: GPUs, checkout state, warnings
./mtr models | tail -1                          # outputs.models: id, kind, task, sizes, measured
./mtr check rf-detr --size small | tail -1      # pauses nothing; downloads weights
# ask the user before the timed run (rule 4)
./mtr run rf-detr --size small | tail -1        # or: ./mtr bench rf-detr --size small
./mtr results | tail -1
./mtr submit rf-detr --dry-run                  # show the user; submit with --yes only on their word (rule 3)
```

**Before you start.**
- `./mtr models` shows each model's CUDA version. A model whose CUDA the driver can't run stops with `UNSUPPORTED`. Tell the user which driver it needs; don't try other models' environments or torch builds.
- Start with the smallest size. `bench` without `--size` measures every size.
- A first `check` downloads several GB. Tell the user before doing it on a metered or small disk.

**Language models.**
- Their first `check` builds llama.cpp, which takes 10–15 minutes and needs `nvcc`, CMake and a C++ compiler. Run it in the background.
- Their `eval` suite (`ifeval`, `gsm8k-cot`) takes about 11 minutes per 8B checkpoint.

## Adding a model

Follow [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/adapters.md](docs/adapters.md). In short:

**An image model:**
1. **Start the directory.** `./mtr new <id>`. It also copies a `meta.toml`: that is the catalog's file, kept by its maintainers. Leave it alone; git ignores it.
2. **Write the adapter from the official README:**
   - its default input size, its loading code and its inference call
   - `TASK`, `SIZES`, `PRECISIONS`, `PACKAGE` and `INPUT`
   - `load()`, `predict()`, `weight_files()` and `to_prediction()`
   
   Pin the official package and torch in `pyproject.toml`. The examples in `tools/measure/examples/` and the other `models/` all work.
3. **Check it.** `./mtr check <id> --size <size>` for each size. The first check creates `uv.lock`; commit it.
4. **Score it, optionally.** Use the task's benchmark: `coco-val2017-bbox` for object detection, `coco-val2017-segm` for instance segmentation. Matting, depth and pose have none yet. Compare with the paper's number and report any gap. Never tune anything to close it.

**A language model (GGUF):**
1. **Start the directory.** `./mtr new <id> --kind llm`.
2. **Name the files.** Put each GGUF file in `WEIGHTS`, with the repository, its 40-hex commit, the file names and the quantization. There's no `pyproject.toml` or lock: llama.cpp runs it.
3. **Check and score.** `./mtr check`, then optionally `./mtr eval` with `ifeval` and `gsm8k-cot`.

**Then, for either kind:**
- **Propose the pull request to the user.** Include the adapter, plus `pyproject.toml`, `uv.lock` and `.python-version` for an image model. Never include records, weights or outputs; git ignores them. Push and open it only with the user's go-ahead.
- **The description** gives the license sources (rule 7) and the `mtr-result` lines of the checks.
- **Patches.** If the official code needed a patch, put it in `models/<id>/fixes/` and say why.
- **Submitting the new model's records** works only after the pull request is merged. Then `git pull`, measure again and submit.

## Files

| Path | What it is |
|---|---|
| `mtr` | The entry point: runs the tool in its uv environment |
| `tools/measure/src/mtr_measure/` | The tool. Start at `main.py` (commands), `cli.py` (check and run), `protocol.py` (constants) and `errors.py` (codes) |
| `tools/measure/src/mtr_measure/schema/` | JSON Schemas of records and submissions. Generated; don't edit |
| `tools/measure/tests/` | `uv run --frozen --project tools/measure pytest tools/measure/tests -q`. No GPU needed |
| `models/<id>/` | One model family: `adapter.py`, `pyproject.toml`, `uv.lock`, optional `fixes/` |
| `timing/` | The fixed timing image per task, with its sha256 |
| `docs/` | Protocol, adapter interface and submitting |
