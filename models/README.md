# models

One directory per model family. The directory name is the model id, a slug such as `rf-detr`.

```
models/<id>/
  adapter.py        # how to load and run the model (docs/adapters.md)
  pyproject.toml    # the model's uv environment: the official package and torch, pinned
  uv.lock           # written by the first `mtr check`, committed
  .python-version   # the Python version uv installs
  fixes/            # optional: patches the official code needed, each explained in a README
```

A language model has only `adapter.py`, naming its GGUF files. The pinned llama.cpp build runs it in the tool's own environment.

What the tool writes stays out of git:

```
  .check/<size>.json                 # the last passed `mtr check`, tied to adapter.py and uv.lock
  runs/<size>.json                   # a run record from `mtr run`
  runs/<size>.times.json             # every timed call behind it
  runs/<size>.log                    # the checks and timings of that run (.failed.log when it failed)
  runs/<size>.submitted.json         # set by `mtr submit`: the inbox's id for the record
  evals/<size>/<benchmark>.json      # accuracy from `mtr eval`; its raw predictions stay next to it
  outputs/<size>/                    # images from `mtr outputs`, or answers.json for a language model
  .venv/                             # the model's environment, free to delete (`mtr clean`)
  meta.toml                          # the catalog's facts about the model, kept by its maintainers
```

**Starting a model.** `mtr new <id>` copies `_template/`, or `_template-llm/` with `--kind llm`.

**Weights stay out of the repository.**
- Torch hub and Hugging Face keep them in their own caches.
- For packages that download into the working directory, point them at `mtr_measure.weights_dir("<id>")`.
- `mtr clean <id> --yes` deletes them.
