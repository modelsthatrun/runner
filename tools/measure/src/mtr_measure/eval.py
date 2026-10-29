"""mtr eval: a checkpoint's accuracy on a standard benchmark.

Usage:
    mtr eval <model> --size <size> --benchmark coco-val2017-bbox

Language models go to llmeval.py, which runs lm-evaluation-harness against llama-server.

It runs the checkpoint over the whole benchmark through the same adapter, uv environment and
precision as a timed run, and scores it with the benchmark's reference implementation. It writes
models/<id>/evals/<size>/<benchmark>.json, an eval record that follows
schema/eval-record.schema.json. Nothing is timed, so it pauses nothing. It checks first that the GPU
has room for the model next to whatever else uses it.

Image benchmarks read predict()'s output through to_prediction(), like the output step, but never
apply the display threshold: a benchmark scores every detection. If the official evaluation calls
the model differently from its README (a lower score threshold, more detections), the adapter says
so with eval_predict(model, x). If the model's class names differ from the benchmark's, the adapter
maps them in EVAL = {"label_map": {"motorbike": "motorcycle"}}. Never change these to raise a score.

`--limit N` scores only the first N images. It is for trying an adapter and writes no record.
"""

from __future__ import annotations

import contextlib
import io
import json
import platform
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import adapter as ad
from . import datasets, machine, protocol, record
from .cli import Log, Parser, finish
from .errors import MeasureError
from .gpu import MIB, Gpu
from .predictions import LABEL_ALIASES, PredictionError
from .runqueue import read_check
from .worker import is_oom


@dataclass(frozen=True)
class Benchmark:
    id: str
    name: str
    dataset: datasets.Dataset
    split: str
    task: str
    iou_type: str  # pycocotools: "bbox" or "segm"
    metric: str


BENCHMARKS = {
    b.id: b
    for b in (
        Benchmark("coco-val2017-bbox", "COCO val2017, box AP", datasets.COCO_VAL2017, "val2017", "object-detection", "bbox", "AP@[.50:.95]"),
        Benchmark("coco-val2017-segm", "COCO val2017, mask AP", datasets.COCO_VAL2017, "val2017", "instance-segmentation", "segm", "AP@[.50:.95]"),
    )
}

# pycocotools' twelve summary numbers, in its order.
COCO_STATS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl")
# COCO keeps the 100 highest-scoring detections per image (maxDets).
MAX_DETS = 100
# Names some models use for COCO classes (predictions.LABEL_ALIASES). A model with other names maps
# them itself in EVAL["label_map"].
COCO_ALIASES = LABEL_ALIASES
# Placeholder labels that some label lists keep for unused class ids. Never a real prediction.
PLACEHOLDERS = {"N/A", "__background__", "background"}


def parse_args(argv: list[str]):
    p = Parser(prog="mtr eval", description="Score one checkpoint on a standard benchmark and write an eval record. Pauses nothing. "
                                            "Image models: coco-val2017-bbox (detection), coco-val2017-segm (instance segmentation); "
                                            "COCO val2017 downloads once, about 1 GB. Language models: ifeval, gsm8k-cot.")
    p.add_argument("model_dir", type=Path)
    p.add_argument("--size", required=True)
    p.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    p.add_argument("--limit", type=int, help="Score only the first N images, to try the adapter. Writes no record")
    p.add_argument("--model-id")
    p.add_argument("--gpu", type=int, default=0, help="NVML device index (default 0)")
    return p.parse_args(argv)


def free_vram_mib(index: int) -> tuple[float, float]:
    gpu = Gpu(index)
    try:
        used, total = gpu.used_bytes(), gpu.total_bytes()
    finally:
        gpu.close()
    return (total - used) / MIB, used / MIB


def label_ids(categories: list[dict], label_map: dict[str, str]) -> dict[str, int]:
    ids = {c["name"]: c["id"] for c in categories}
    for alias, name in {**COCO_ALIASES, **label_map}.items():
        if name in ids:
            ids[alias] = ids[name]
    return ids


def coco_detections(pred: Any, image_id: int, ids: dict[str, int], iou_type: str, unknown: set[str]) -> list[dict]:
    """One image's predictions as COCO result entries, highest scores first, at most MAX_DETS."""
    if pred.scores is None:
        raise MeasureError("EVAL", "The predictions have no scores. COCO AP ranks detections by score")
    order = sorted(range(len(pred)), key=lambda i: -float(pred.scores[i]))
    out = []
    for i in order:
        label = pred.labels[i]
        if label in PLACEHOLDERS:
            continue
        if label not in ids:
            unknown.add(label)
            continue
        entry = {"image_id": image_id, "category_id": ids[label], "score": round(float(pred.scores[i]), 5)}
        if iou_type == "bbox":
            x1, y1, x2, y2 = (float(v) for v in pred.xyxy[i])
            entry["bbox"] = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
        else:
            import numpy as np
            from pycocotools import mask as mask_util

            rle = mask_util.encode(np.asfortranarray(pred.masks[i].astype(np.uint8)))
            rle["counts"] = rle["counts"].decode()
            entry["segmentation"] = rle
        out.append(entry)
        if len(out) == MAX_DETS:
            break
    return out


def main(argv: list[str], log: Log) -> int:
    if argv and not argv[0].startswith("-") and (Path(argv[0]) / "adapter.py").is_file() \
            and ad.kind_of_file(Path(argv[0]) / "adapter.py") == "llm":
        from . import llmeval

        return llmeval.main(argv, log)
    args = parse_args(argv)
    machine.pin_gpu(args.gpu)
    bench = BENCHMARKS[args.benchmark]
    model_dir = args.model_dir.resolve()
    if not (model_dir / "adapter.py").is_file():
        raise MeasureError("USAGE", f"No adapter.py in {model_dir}")
    model_id = args.model_id or model_dir.name
    out = model_dir / "evals" / args.size / f"{bench.id}.json"
    log_path = out.with_suffix(".log") if not args.limit else out.with_name(f"{bench.id}.limit.log")
    try:
        return evaluate(args, bench, model_dir, model_id, out, log_path, log)
    except MeasureError as e:
        log(f"FAILED: {e}")
        return finish(log, log_path, e.code, str(e))


def evaluate(args, bench: Benchmark, model_dir: Path, model_id: str, out: Path, log_path: Path, log: Log) -> int:
    log(f"mtr-measure eval {model_id} / {args.size} on {bench.name}" + (f" (first {args.limit} images only, no record)" if args.limit else ""))
    log(f"command: {shlex.join(sys.argv)}", echo=False)
    check = read_check(model_dir, args.size)
    free, used = free_vram_mib(args.gpu)
    need = (check.get("approx_vram_mib") or 0) + protocol.EVAL_VRAM_MARGIN_MIB
    log(f"gpu memory: {free:.0f} MiB free ({used:.0f} MiB in use by other programs); this model needs about {need} MiB")
    if free < need:
        raise MeasureError("VRAM_LOW", f"Only {free:.0f} MiB of GPU memory is free and the model needs about {need} MiB")

    base = datasets.ensure(bench.dataset, log)
    ann = base / "annotations" / "instances_val2017.json"
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann))
    image_ids = sorted(gt.getImgIds())
    if args.limit:
        image_ids = image_ids[: args.limit]

    a = ad.load_adapter(model_dir / "adapter.py")
    ad.check_size(a, args.size)
    precision = ad.choose_precision(list(a.PRECISIONS))
    settings_extra = dict(getattr(a, "EVAL", {}) or {})
    label_map = dict(settings_extra.pop("label_map", {}) or {})
    if settings_extra:
        raise MeasureError("ADAPTER_INTERFACE", f"EVAL only takes label_map, not {', '.join(settings_extra)}")
    run = getattr(a, "eval_predict", a.predict)
    ids = label_ids(gt.loadCats(gt.getCatIds()), label_map)
    log(f"package: {ad.package(a)}, precision {precision}, {'eval_predict()' if run is not a.predict else 'predict()'}")

    t0 = time.perf_counter()
    model = a.load(args.size, precision)
    results: list[dict] = []
    unknown: set[str] = set()
    for k, image_id in enumerate(image_ids, 1):
        info = gt.loadImgs(image_id)[0]
        x = ad.prepare_input(a, [base / bench.split / info["file_name"]])
        try:
            pred = ad.to_prediction(a, model, x, run(model, x), bench.task)
        except PredictionError as e:
            raise MeasureError("ADAPTER_RUNTIME", f"to_prediction() on {info['file_name']}: {e}") from e
        except Exception as e:
            if is_oom(e):
                raise MeasureError("OOM", f"Out of GPU memory on {info['file_name']}: {e}") from e
            raise
        results += coco_detections(pred, image_id, ids, bench.iou_type, unknown)
        if k % 500 == 0:
            log(f"  {k} of {len(image_ids)} images, {time.perf_counter() - t0:.0f} s")
    if unknown:
        raise MeasureError("EVAL", f"The model predicts labels that aren't COCO classes: {', '.join(sorted(unknown))}. "
                                   "If they are COCO classes under other names, map them in the adapter's EVAL['label_map']")
    if not results:
        raise MeasureError("EVAL", "The model made no detections on the benchmark")

    preds_path = out.with_name(f"{bench.id}.predictions.json" if not args.limit else f"{bench.id}.limit.predictions.json")
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_path.write_text(json.dumps(results))
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(str(preds_path))
    ev = COCOeval(gt, dt, bench.iou_type)
    ev.params.imgIds = image_ids
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    for line in buf.getvalue().strip().splitlines():
        if line.lstrip().startswith("Average"):
            log(line)
    metrics = {name: round(float(v) * 100, 2) for name, v in zip(COCO_STATS, ev.stats)}
    duration = time.perf_counter() - t0
    log(f"{bench.metric}: {metrics['AP']} on {len(image_ids)} images in {duration:.0f} s")

    numbers = {"value": metrics["AP"], "metrics": metrics, "images": len(image_ids)}
    if args.limit:
        return finish(log, log_path, "OK", f"{bench.metric} {metrics['AP']} on the first {len(image_ids)} images only: no record written", numbers)

    import importlib.metadata

    import torch

    gpu = Gpu(args.gpu)
    try:
        gpu_name, driver = gpu.name(), gpu.driver()
    finally:
        gpu.close()
    rec = {
        "kind": "image",
        "model": model_id,
        "size": args.size,
        "weights": record.weight_entries([Path(p) for p in a.weight_files(args.size)]),
        "benchmark": {"id": bench.id, "name": bench.name, "split": bench.split, "n": len(image_ids),
                      "metric": bench.metric, "data_sha256": record.sha256_file(ann)},
        "value": metrics["AP"],
        "metrics": metrics,
        "settings": {
            "precision": precision,
            "max_dets": MAX_DETS,
            "score_threshold": "none applied by the tool; the model's own post-processing only",
            "call": "eval_predict()" if run is not a.predict else "predict()",
            "label_map": label_map,
            "evaluator": f"pycocotools=={importlib.metadata.version('pycocotools')}",
        },
        "env": {
            "gpu": gpu_name, "driver": driver, "cuda": str(torch.version.cuda), "torch": torch.__version__,
            "python": platform.python_version(), "package": ad.package(a),
        },
        "protocol": protocol.PROTOCOL_VERSION,
        "evaluated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_s": round(duration),
        "predictions_sha256": record.sha256_file(preds_path),
        "provenance": machine.provenance(model_dir, None),
    }
    problems = record.validate(rec, "eval-record")
    for prob in problems:
        log(f"SCHEMA: {prob}")
    if problems:
        return finish(log, log_path, "SCHEMA", f"The eval record fails the schema: {problems[0]}", numbers)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    log(f"wrote {out} (the predictions, {preds_path.name}, stay here as evidence and are never pushed)")
    return finish(log, log_path, "OK", f"{model_id} / {args.size}: {bench.metric} {metrics['AP']} on {bench.name}",
                  {"record": str(out), **numbers})
