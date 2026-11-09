"""The output step for text-to-image models: one checkpoint makes an image for every prompt of its task's set.

    mtr outputs <id> --size <size>

It loads the pipeline through the adapter, in the model's environment and at the precision `mtr run` uses,
and calls predict() once per prompt of prompts/<task>.toml with the set's seed and the adapter's GENERATION
settings, exactly as a timed call is made. It writes models/<id>/outputs/<size>/<prompt-id>.webp and
generations.json (schema/generations.schema.json), plus outputs/<size>.log. Failures are flagged by hand
afterwards, against each prompt's check.

The same seed doesn't give the same pixels on another GPU, driver or library version, or even in another
process, so nothing compares images by their hashes. The hashes only tie each file to this run.

Nothing is timed, so nothing is paused.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import sys
from datetime import datetime
from pathlib import Path

from . import adapter as ad
from . import generation, prompts, record, render
from .cli import Log
from .errors import MeasureError


def run(model_dir: Path, model_id: str, size: str, task: str, log: Log, out: Path | None = None) -> int:
    s = prompts.load(task)
    a = ad.load_adapter(model_dir / "adapter.py")
    ad.check_size(a, size)
    precision = ad.choose_precision(list(a.PRECISIONS))
    g = generation.settings(a, size)
    out_dir = (out or model_dir / "outputs" / size).resolve()
    log(f"mtr outputs {model_id} / {size}: {len(s.prompts)} prompts of the {task} set, version {s.version}")
    log(f"command: {shlex.join(sys.argv)}", echo=False)
    log(f"package: {ad.package(a)}, precision {precision}, {g['resolution']}, {g['steps']} steps, guidance {g['guidance']:g}, "
        f"seed {s.seed} on {g['generator']}")

    import torch

    if not torch.cuda.is_available():
        raise MeasureError("FAILED", "torch can't see a CUDA device")
    if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise MeasureError("FAILED", "This GPU only emulates bf16, and the protocol doesn't run an emulation")
    model = a.load(size, precision)
    sched = generation.scheduler(a, model)

    # Make everything before writing anything, so a failure leaves the previous set in place.
    made: list[tuple[str, bytes, tuple[int, int]]] = []
    for p in s.prompts:
        x = generation.make(a, size, p.prompt, s.seed)
        img = generation.check_image(a.predict(model, x), x)
        made.append((p.id, render.encode_webp(img), img.size))
        log(f"{p.id}: {img.size[0]}x{img.size[1]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for pid, data, (w, h) in made:
        tmp = out_dir / f".{pid}.webp.tmp"
        tmp.write_bytes(data)
        tmp.replace(out_dir / f"{pid}.webp")
        images.append({"prompt": pid, "file": f"{pid}.webp", "sha256": hashlib.sha256(data).hexdigest(), "width": w, "height": h})
    doc = {
        "kind": "generations",
        "task": task,
        "model": model_id,
        "size": size,
        "weights": record.weight_entries([Path(f) for f in a.weight_files(size)]),
        "prompt_set": {"task": task, "version": s.version, "sha256": s.sha256},
        "precision": precision,
        "resolution": g["resolution"],
        "settings": generation.recorded(generation.make(a, size, "", s.seed), sched),
        "env": {"cuda": str(torch.version.cuda), "torch": torch.__version__, "python": platform.python_version(), "package": ad.package(a)},
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "images": images,
    }
    problems = record.validate(doc, "generations")
    if problems:
        raise MeasureError("SCHEMA", f"generations.json fails the schema: {problems[0]}")
    path = out_dir / "generations.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    log(f"wrote {len(images)} image(s) and {path.name} to {out_dir}")
    ids = {i["prompt"] for i in images}
    for stale in sorted(f for f in out_dir.iterdir() if f.is_file() and f.suffix == ".webp" and f.stem not in ids):
        log(f"WARNING: {stale.name} is not in this prompt set. Delete it")
    log("next: look at each image against its prompt's check and flag the failures")
    log_path = out_dir.parent / f"{out_dir.name}.log"
    log_path.write_text(log.text())
    print(f"log: {log_path}")
    return 0
