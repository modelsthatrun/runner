"""mtr outputs: run one checkpoint on a task's test images and render the outputs as WebP.

Usage: mtr outputs <model> --size <size> --images <dir>

It loads the model through the same adapter, uv environment and precision as `mtr run`, applies the
task's fixed score threshold (predictions.THRESHOLDS) and draws every output with the shared code in
render.py. It writes models/<id>/outputs/<size>/<image-id>.webp, plus outputs/<size>.log with the
inputs' sha256 and what each image kept. Nothing is timed, so it pauses nothing.

The images are the task's shared test set: the catalog's maintainers fetch it and pass it with
--images. Each file's name without its suffix is the image id. Outputs of images under a `_dev`
directory never go into the model's outputs: they go to <that _dev directory>/outputs/<id>/<size>/.

Language models answer their task's prompt set instead (llmoutputs.py), and need no --images.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import adapter as ad
from . import machine, paths, predictions, record, render
from .cli import Log
from .errors import MeasureError

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DEV = re.compile(r"(^|[\\/])_dev([\\/]|$)")  # the same rule as isDevPath() in packages/schema


class OutputsError(Exception):
    pass


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mtr outputs", description="Render one checkpoint's outputs on a task's test images.")
    p.add_argument("model_dir", type=Path, help="The model: an id such as rf-detr, or its directory")
    p.add_argument("--size", required=True, help="Checkpoint size, as in the adapter's SIZES")
    p.add_argument("--images", type=Path, help="Directory of the task's test images, named <image-id>.jpg. Required for image models")
    p.add_argument("--task", help="Task id (default: the adapter's TASK)")
    p.add_argument("--model-id", help="Model id (default: the directory name)")
    p.add_argument("--out", type=Path, help="Output directory (default: <model_dir>/outputs/<size>)")
    p.add_argument("--gpu", type=int, default=0, help="NVML device index (default 0)")
    return p.parse_args(argv)


def is_dev(path: Path) -> bool:
    return bool(DEV.search(str(path.resolve())))


def dev_root(path: Path) -> Path | None:
    """The `_dev` directory a file sits under, if any."""
    for d in path.resolve().parents:
        if d.name == "_dev":
            return d
    return None


def read_task(model_dir: Path) -> str:
    task = ad.task_of_file(model_dir / "adapter.py")
    if not task:
        raise OutputsError(f"{model_dir / 'adapter.py'} declares no TASK. Add it, or pass --task")
    return task


def list_images(directory: Path) -> list[tuple[str, Path]]:
    """(image id, file) for every image in a directory. The id is the file name without suffix."""
    if not directory.is_dir():
        raise OutputsError(f"Not a directory: {directory}")
    images: dict[str, Path] = {}
    for f in sorted(directory.iterdir()):
        if not f.is_file() or f.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not SLUG.match(f.stem):
            raise OutputsError(f"{f.name}: the file name must be <image-id>{f.suffix}, with a lowercase-hyphen id")
        if f.stem in images:
            raise OutputsError(f"Two files for image {f.stem}: {images[f.stem].name} and {f.name}")
        images[f.stem] = f
    if not images:
        raise OutputsError(f"No images ({', '.join(IMAGE_SUFFIXES)}) in {directory}")
    return list(images.items())


def output_dir(args: argparse.Namespace, model_dir: Path, model_id: str, dev: Path | None) -> Path:
    if args.out:
        out = args.out.resolve()
        if dev and not is_dev(out):
            raise OutputsError("Outputs of images under a _dev directory must stay under one, so they never reach a model's outputs. Drop --out")
        return out
    if dev:
        return dev / "outputs" / model_id / args.size
    return model_dir / "outputs" / args.size


def summary(before: Any, after: Any, task: str) -> str:
    # Mattes and depth maps are one map per image, with nothing to count.
    if isinstance(after, predictions.Alpha):
        return f"alpha matte, {100 * float((after.alpha >= 0.5).mean()):.1f}% foreground"
    if isinstance(after, predictions.Depth):
        return f"depth map, {100 * float(np.isfinite(after.depth).mean()):.1f}% of pixels valid"
    labels = getattr(after, "labels", None)
    counts = Counter(["person"] * len(after) if isinstance(after, predictions.Keypoints) else labels or [])
    found = ", ".join(f"{label} ×{n}" if n > 1 else label for label, n in counts.most_common()) or "nothing"
    if getattr(before, "scores", None) is None:
        return f"{len(after)}: {found}"
    return f"{len(after)} of {len(before)} at score ≥ {predictions.THRESHOLDS[task]}: {found}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    machine.pin_gpu(args.gpu)
    args.model_dir = paths.model_dir(args.model_dir)
    log = Log()
    try:
        model_dir = args.model_dir.resolve()
        if (model_dir / "adapter.py").is_file() and ad.kind_of_file(model_dir / "adapter.py") == "llm":
            from . import llmoutputs

            return llmoutputs.run(model_dir, args.model_id or model_dir.name, args.size, args.task or read_task_or(model_dir, "chat"), log, args.gpu)
        return run(args, log)
    except (OutputsError, ad.AdapterError, predictions.PredictionError, NotImplementedError) as e:
        log(f"FAILED: {e}")
        return 1
    except MeasureError as e:
        log(f"FAILED: {e}")
        return 1


def read_task_or(model_dir: Path, default: str) -> str:
    """The adapter's TASK, or the default when it declares none."""
    try:
        return read_task(model_dir)
    except OutputsError:
        return default


def run(args: argparse.Namespace, log: Log) -> int:
    model_dir = args.model_dir.resolve()
    model_id = args.model_id or model_dir.name
    adapter_path = model_dir / "adapter.py"
    if not adapter_path.is_file():
        raise OutputsError(f"No adapter.py in {model_dir}. Start one with `mtr new <id>`")
    task = args.task or read_task(model_dir)
    kind = predictions.task_type(task)
    render.drawer(kind)  # fail before loading anything if this type can't be drawn yet

    log(f"mtr outputs {model_id} / {args.size}, task {task}")
    log(f"command: {shlex.join(sys.argv)}", echo=False)
    if not args.images:
        raise OutputsError("Pass --images <dir>: the task's test images, named <image-id>.jpg")
    images = list_images(args.images.resolve())
    log(f"images: {len(images)} from {args.images}")
    dev = next((d for d in (dev_root(f) for _, f in images) if d), None)
    out_dir = output_dir(args, model_dir, model_id, dev)
    if dev:
        log(f"development images under {dev}: their outputs stay there, out of the model's outputs")

    a = ad.load_adapter(adapter_path)
    ad.check_size(a, args.size)
    precision = ad.choose_precision(list(a.PRECISIONS))
    log(f"package: {ad.package(a)}, precision {precision}")
    threshold = predictions.THRESHOLDS.get(task)
    log(f"threshold: {threshold if threshold is not None else 'none set'} (predictions.THRESHOLDS)")
    model = a.load(args.size, precision)

    # Render everything before writing anything, so a failure leaves the previous set in place.
    rendered: list[tuple[str, bytes]] = []
    for image_id, f in images:
        log(f"input {image_id}: {f.name} sha256 {record.sha256_file(f)}", echo=False)
        x = ad.prepare_input(a, [f])
        out = a.predict(model, x)
        try:
            pred = ad.to_prediction(a, model, x, out, task)
            kept = predictions.canonical_labels(predictions.apply_threshold(pred, task))
            with Image.open(f) as im:
                data = render.encode_webp(render.render(im.convert("RGB"), kept))
        except (ad.AdapterError, predictions.PredictionError) as e:
            raise OutputsError(f"{image_id}: {e}") from None
        rendered.append((image_id, data))
        log(f"{image_id}: {summary(pred, kept, task)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for image_id, data in rendered:
        tmp_file = out_dir / f".{image_id}.webp.tmp"
        tmp_file.write_bytes(data)
        tmp_file.replace(out_dir / f"{image_id}.webp")
    log(f"wrote {len(rendered)} output(s) to {out_dir}")
    ids = {image_id for image_id, _ in rendered}
    for stale in sorted(p for p in out_dir.iterdir() if p.is_file() and p.stem not in ids and not p.name.startswith(".")):
        log(f"WARNING: {stale.name} is not in this test set. Delete it")
    if not dev:
        log("next: look at the outputs and flag the failures")
    # Next to the size directory, not in it, so it never counts as an output.
    log_path = out_dir.parent / f"{out_dir.name}.log"
    log_path.write_text(log.text())
    print(f"log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
