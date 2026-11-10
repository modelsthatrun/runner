"""mtr eval for text-to-image models (KIND = "generation"): GenEval.

Usage:
    mtr eval <id> --size <size> --benchmark geneval

Two steps:

1. **Images**, in the model's own environment. For each of GenEval's 553 prompts, four calls of predict(), made
   by the tool as for a timed call (generation.py): the prompt as GenEval wrote it, the adapter's GENERATION
   settings, and seeds 42, 43, 44 and 45. No prompt rewriter and no negative prompt. They are written as PNG in
   the layout GenEval's evaluate_images.py reads: models/<id>/evals/<size>/geneval/<NNNNN>/samples/<k>.png, with
   the prompt's metadata.jsonl. A run that stops resumes: images already made with the same settings are kept
   (geneval/run.json says which), and images made otherwise are refused.
2. **Scores**, in GenEval's own environment (tools/measure/geneval/: Python 3.9, torch 1.12.1 with CUDA 11.3,
   mmcv-full 1.7.1, mmdetection 2.28.2). The tool runs GenEval's evaluate_images.py, unchanged, at a pinned
   commit and with its default options: Mask2Former (Swin-S) finds the objects and CLIP ViT-L/14 names their
   colors. It then runs GenEval's summary_scores.py and checks that its overall score is the one the tool
   computed from the same results.

It writes models/<id>/evals/<size>/geneval.json, an eval record whose value is GenEval's overall score, 0-1 as
GenEval reports it. The images and GenEval's per-image results (geneval.results.jsonl) stay next to it as
evidence and are never published or pushed. Nothing is timed, so nothing is paused; free GPU memory is checked
before each step. `--limit N` makes and scores only the first N prompts, in geneval.limit/, and writes no record.

`--pause` is for a model that only fits on the GPU with the machine's pause hooks (machine.toml, desktop.py)
stopped, as for a timed run. The hooks then hold while the images are made, which takes hours, and the free-memory
check before them leaves EVAL_VRAM_MARGIN_PAUSED_MIB instead of EVAL_VRAM_MARGIN_MIB. Scoring runs after they
start again. The pause stops programs such as a streaming host, so use it only when whoever uses the machine
agrees. The margins decide only whether the run starts, never a score.

GenEval's files, the detector's checkpoint and CLIP's weights download once, each checked by its sha256.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import adapter as ad
from . import datasets, desktop, generation, machine, paths, protocol, record
from .cli import Log, Parser, finish
from .errors import MeasureError
from .gpu import MIB, Gpu
from .runqueue import read_check
from .worker import is_oom


@dataclass(frozen=True)
class Asset:
    url: str
    bytes: int
    sha256: str


COMMIT = protocol.GENEVAL_COMMIT
RAW = f"https://raw.githubusercontent.com/djghosh13/geneval/{COMMIT}/"
# GenEval's files the tool uses, at COMMIT: (bytes, sha256). Read on 2026-11-10.
FILES = {
    "LICENSE": (1069, "e6082bfcbdf460dd19f8c55cc261055414df2a278768b4991c5382d65be2b52a"),
    "evaluation/evaluate_images.py": (11597, "c0fdeb8315f7d3ed72b56f466652afba5216f854109c6eeea339f9982474ea46"),
    "evaluation/summary_scores.py": (1179, "302212beb80359b03fde980f4a71e76e44e41969bd917af6345310a715761aae"),
    "evaluation/object_names.txt": (642, "608f6a0e5c8ca1c7a92b818430141fe268c85b7930b50f4160f1d65893b5bacd"),
    "prompts/evaluation_metadata.jsonl": (85209, "5c48e0813e812e3c373fa5c8ed07a8f0a483be30272b4427b0559c8048e67c13"),
}
# The object detector GenEval's evaluation/download_models.sh fetches (MMDetection's model zoo, Apache-2.0),
# saved under the name evaluate_images.py looks for.
DETECTOR_NAME = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
DETECTOR = Asset(
    "https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/"
    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth",
    275_630_991, "743b7d99015f1224c6d57fd4b14d04b15cc8ec72ae7ee7831e7c71d8873b7a54")
# CLIP ViT-L/14 by OpenAI (MIT), the color classifier: the URL open_clip 2.20.0 downloads it from, and the sha256
# that URL names. The tool fetches it into open_clip's own cache first, so a slow download can resume.
CLIP = Asset(
    "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    932_768_134, "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836")
CLIP_CACHE = Path("~/.cache/clip").expanduser()
# GenEval's six tasks, by the tags in its prompts, in its order.
TAGS = ("single_object", "two_object", "counting", "colors", "position", "color_attr")
# The evaluator's peak GPU memory on 1024x1024 images: 6573 MiB over the desktop's, measured with nvidia-smi on an
# RTX 4090 on 2026-11-10, rounded up.
SCORE_VRAM_MIB = 7168
OPTIONS = ("none: evaluate_images.py's defaults, threshold 0.3, counting_threshold 0.9, max_objects 16, "
           "max_overlap 1.0, position_threshold 0.1, gray background, crop")
SEEDING = ("one generator per image, seeded 42, 43, 44 and 45 for the four images of every prompt, on the adapter's "
           "generator device. GenEval's own script seeds 42 once per prompt and draws the four in a row")


def parse_args(argv: list[str]):
    p = Parser(prog="mtr eval", description="Score one text-to-image checkpoint on GenEval: make 553 x 4 images, "
                                            "then score them with GenEval's own evaluator. Pauses nothing unless --pause.")
    p.add_argument("model_dir", type=Path)
    p.add_argument("--size", required=True)
    p.add_argument("--benchmark", required=True, choices=sorted(protocol.GEN_BENCHMARKS))
    p.add_argument("--limit", type=int, help="Only the first N prompts, to try it. Writes no record")
    p.add_argument("--model-id")
    p.add_argument("--gpu", type=int, default=0, help="NVML device index (default 0)")
    p.add_argument("--pause", action="store_true",
                   help="Stop the machine's pause hooks (machine.toml) while images are made, for a model that only fits "
                        "with them stopped. Takes hours: use it only when whoever uses the machine agrees")
    return p.parse_args(argv)


# --- GenEval's files, checkpoint and weights ------------------------------------------------------


def home() -> Path:
    return datasets.root() / "geneval"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_checked(asset: Asset, dest: Path, say) -> Path:
    """dest, downloading it first if needed; refused unless its sha256 is the expected one."""
    if dest.is_file() and dest.stat().st_size == asset.bytes and record.sha256_file(dest) == asset.sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    say(f"downloading {asset.url} ({asset.bytes / 2**20:.0f} MiB, once)")
    datasets.fetch(asset.url, dest, asset.bytes, say)
    got = record.sha256_file(dest)
    if got != asset.sha256:
        dest.unlink()
        raise MeasureError("DATASET", f"{dest.name} has sha256 {got[:12]}…, expected {asset.sha256[:12]}…. Deleted it")
    return dest


def ensure_files(say) -> Path:
    """GenEval's own files at COMMIT, each checked by its sha256."""
    base = home() / COMMIT
    for rel, (size, sha) in FILES.items():
        fetch_checked(Asset(RAW + rel, size, sha), base / rel, say)
    return base


def ensure_weights(say) -> Path:
    """The detector's checkpoint (returned: the directory evaluate_images.py's --model-path takes) and CLIP's weights."""
    fetch_checked(DETECTOR, home() / "detector" / f"{DETECTOR_NAME}.pth", say)
    fetch_checked(CLIP, CLIP_CACHE / "ViT-L-14.pt", say)
    return home() / "detector"


def prompts(base: Path) -> list[dict]:
    return [json.loads(line) for line in (base / "prompts" / "evaluation_metadata.jsonl").read_text().splitlines() if line.strip()]


# --- the evaluator's environment ------------------------------------------------------------------


def project() -> Path:
    return paths.tool_dir() / "geneval"


def uv() -> str:
    return os.environ.get("UV") or "uv"


def nvrtc(log_file: Path) -> Path:
    """NVRTC 11.8 from NVIDIA's wheel in GenEval's environment, which the evaluator's Python preloads.

    torch 1.12 compiles a few kernels when they're first used, such as the integer product of feature-map shapes in
    Mask2Former's pixel decoder, with the NVRTC it bundles (CUDA 11.3). It asks for the GPU's own architecture, and
    NVRTC 11.3 doesn't know sm_89 (RTX 40) and later, so the evaluator stops there. NVRTC 11.8 does. The kernels'
    source is torch's own, so only the compiler that turns it into GPU code changes, never the scoring.
    """
    with log_file.open("a") as f:
        rc = subprocess.run([uv(), "sync", "--frozen", "--project", str(project())], stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise MeasureError("EVAL", f"Couldn't install GenEval's environment (tools/measure/geneval/): see {log_file}")
    found = sorted((project() / ".venv").glob("lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.*"))
    if not found:
        raise MeasureError("EVAL", "GenEval's environment has no NVRTC from nvidia-cuda-nvrtc-cu11")
    return found[0]


def in_evaluator(args: list[str], preload: Path | None = None) -> list[str]:
    """A command in GenEval's environment, from its committed lock, isolated from this process's Python. With
    `preload`, only that Python process loads it first, not uv."""
    env = ["env", f"LD_PRELOAD={preload}"] if preload else []
    return [uv(), "run", "--frozen", "--project", str(project()), *env, "python", "-I", *args]


EVALUATOR_INFO = f"""
import json, os, importlib.metadata as md
info = {{}}
try:
    import torch, mmdet
    info = {{
        "python": __import__("platform").python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "nvrtc": md.version("nvidia-cuda-nvrtc-cu11") + " (preloaded)",
        "mmcv-full": md.version("mmcv-full"), "mmdet": mmdet.__version__, "open-clip-torch": md.version("open-clip-torch"),
        "clip-benchmark": md.version("clip-benchmark"),
        "config": os.path.join(os.path.dirname(mmdet.__file__), ".mim", "configs", "mask2former", "{DETECTOR_NAME}.py"),
    }}
    major, minor = torch.cuda.get_device_capability(0)
    info["arch"] = f"sm_{{major}}{{minor}}"
    # What the evaluator needs of the GPU: a kernel torch ships (a convolution), and one it compiles at first use.
    x = torch.ones(1, 3, 8, 8, device="cuda")
    torch.nn.functional.conv2d(x, torch.ones(4, 3, 3, 3, device="cuda"))
    assert torch.tensor([[64, 64], [32, 32]], device="cuda").prod(1).tolist() == [4096, 1024]
    maps = open("/proc/self/maps").read()
    info["nvrtc_loaded"] = sorted({{l.split()[-1].rsplit("/", 1)[-1] for l in maps.splitlines() if "nvrtc" in l}})
    info["gpu_ok"] = True
except Exception as e:
    info["gpu_error"] = f"{{type(e).__name__}}: {{str(e).strip().splitlines()[-1] if str(e).strip() else ''}}"
print(json.dumps(info))
"""


def evaluator_info(log_file: Path) -> dict:
    """The evaluator environment's versions and the detector's config file, which mmdetection's wheel keeps under
    mmdet/.mim/configs (GenEval's default path assumes a source checkout). It also tries the GPU the way the evaluator
    will use it, so a GPU it can't run on stops the eval before any image is made."""
    preload = nvrtc(log_file)
    with log_file.open("a") as f:
        proc = subprocess.run(in_evaluator(["-c", EVALUATOR_INFO], preload), stdout=subprocess.PIPE, stderr=f, text=True)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    if proc.returncode != 0 or not lines:
        raise MeasureError("EVAL", f"GenEval's environment (tools/measure/geneval/) didn't start: see {log_file}")
    info = json.loads(lines[-1])
    if not info.pop("gpu_ok", False):
        raise MeasureError("UNSUPPORTED", f"GenEval's environment (torch 1.12.1 with CUDA 11.3) can't run on this GPU"
                                          f"{' (' + info['arch'] + ')' if info.get('arch') else ''}: {info.get('gpu_error')}. "
                                          "It has kernels up to sm_86, which run on sm_8x GPUs such as RTX 30 and RTX 40 cards")
    if not any(n.startswith("libnvrtc.so") for n in info.pop("nvrtc_loaded", [])):
        raise MeasureError("EVAL", "The evaluator didn't load the preloaded NVRTC 11.8")
    if not Path(info["config"]).is_file():
        raise MeasureError("EVAL", f"mmdetection has no {info['config']}")
    info["preload"] = str(preload)
    return info


# --- images ---------------------------------------------------------------------------------------


def image_dir(out: Path, limit: int | None) -> Path:
    return out.with_name("geneval.limit" if limit else "geneval")


def sample_path(images: Path, prompt_index: int, k: int) -> Path:
    return images / f"{prompt_index:05d}" / "samples" / f"{k:04d}.png"


def missing(images: Path, n_prompts: int, n_images: int) -> list[tuple[int, int]]:
    return [(i, k) for i in range(n_prompts) for k in range(n_images) if not sample_path(images, i, k).is_file()]


def settings_of(a: Any, model_dir: Path, size: str, precision: str, seeds: list[int], prompts_sha: str, weights: list[dict]) -> dict:
    """What the images depend on. Images made under other settings are never mixed into one score."""
    g = generation.settings(a, size)
    return {"size": size, "weights": [w["sha256"] for w in weights], "precision": precision, "package": ad.package(a),
            "resolution": g["resolution"], "steps": g["steps"], "guidance": float(g["guidance"]), "generator": g["generator"],
            "seeds": seeds, "prompts_sha256": prompts_sha, **record.fingerprint(model_dir)}


def read_state(images: Path) -> dict | None:
    f = images / "run.json"
    return json.loads(f.read_text()) if f.is_file() else None


def write_state(images: Path, state: dict) -> None:
    f = images / "run.json"
    tmp = f.with_name(".run.json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(f)


def save_png(img: Any, dest: Path) -> None:
    """Written whole or not at all: a stopped run never leaves a half-written image behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp")
    img.save(tmp, format="PNG")
    tmp.replace(dest)


def vram_needed(check: dict, pause: bool) -> int:
    """Free MiB the image step needs before it starts: what `check` saw the model use, and a margin."""
    margin = protocol.EVAL_VRAM_MARGIN_PAUSED_MIB if pause else protocol.EVAL_VRAM_MARGIN_MIB
    return (check.get("approx_vram_mib") or 0) + margin


def make_images(a: Any, model_dir: Path, size: str, precision: str, rows: list[dict], seeds: list[int], images: Path,
                state: dict, gpu: int, log: Log, pause: bool = False) -> None:
    todo = missing(images, len(rows), len(seeds))
    if not todo:
        log(f"all {len(rows) * len(seeds)} images are made already")
        return
    for stale in images.glob("*/samples/.*.tmp"):
        stale.unlink()
    check = read_check(model_dir, size)
    # The hooks hold for the whole image step and start again on the way out, whatever happens (desktop.paused).
    with desktop.paused(log) if pause else nullcontext():
        free, used = free_vram_mib(gpu)
        need = vram_needed(check, pause)
        log(f"gpu memory: {free:.0f} MiB free ({used:.0f} MiB in use by other programs); the model needs about {need} MiB"
            + (" with the pause hooks stopped" if pause else ""))
        if free < need:
            raise MeasureError("VRAM_LOW", f"Only {free:.0f} MiB of GPU memory is free and the model needs about {need} MiB")
        _make(a, size, precision, rows, seeds, images, state, todo, log, pause)


def _make(a: Any, size: str, precision: str, rows: list[dict], seeds: list[int], images: Path, state: dict,
          todo: list, log: Log, pause: bool) -> None:
    cuda_ready(precision)
    log(f"making {len(todo)} of {len(rows) * len(seeds)} images ({len(rows) * len(seeds) - len(todo)} made already)")
    model = a.load(size, precision)
    state["scheduler"] = generation.scheduler(a, model)
    write_state(images, state)
    pending = {i for i, _ in todo}
    t0 = last = time.perf_counter()
    done = 0
    try:
        for i, row in enumerate(rows):
            if i not in pending:
                continue
            meta = images / f"{i:05d}" / "metadata.jsonl"
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text(json.dumps(row))  # one line, as GenEval's own generation script writes it
            for k, seed in enumerate(seeds):
                dest = sample_path(images, i, k)
                if dest.is_file():
                    continue
                x = generation.make(a, size, row["prompt"], seed)
                try:
                    img = generation.check_image(a.predict(model, x), x)
                except Exception as e:
                    if is_oom(e):
                        raise MeasureError("OOM", f"Out of GPU memory on prompt {i}: {e}") from e
                    raise
                save_png(img, dest)
                done += 1
                now = time.perf_counter()
                if now - last >= 60 or done == len(todo):
                    if pause:
                        desktop.extend_fallback()  # so Sunshine's fallback timer doesn't start it in the middle
                    state["generation_s"] = round(state.get("generation_s", 0) + now - last, 1)
                    write_state(images, state)
                    last = now
                    left = (now - t0) / done * (len(todo) - done)
                    log(f"  {done} of {len(todo)} images, {now - t0:.0f} s, about {left / 60:.0f} min left")
    finally:
        now = time.perf_counter()
        state["generation_s"] = round(state.get("generation_s", 0) + now - last, 1)
        write_state(images, state)
        del model
        release()


def cuda_ready(precision: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise MeasureError("FAILED", "torch can't see a CUDA device")
    if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise MeasureError("FAILED", "This GPU only emulates bf16, and the protocol doesn't run an emulation")


def release() -> None:
    """Give the pipeline's GPU memory back before GenEval's evaluator starts."""
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def torch_env() -> dict:
    import torch

    return {"cuda": str(torch.version.cuda), "torch": torch.__version__}


def check_layout(images: Path, rows: list[dict], n_images: int) -> None:
    """What evaluate_images.py will read: one folder per prompt with its metadata, and every image."""
    gaps = missing(images, len(rows), n_images)
    if gaps:
        raise MeasureError("EVAL", f"{len(gaps)} images are missing, e.g. {sample_path(images, *gaps[0])}")
    for i, row in enumerate(rows):
        if json.loads((images / f"{i:05d}" / "metadata.jsonl").read_text()) != row:
            raise MeasureError("EVAL", f"{images / f'{i:05d}' / 'metadata.jsonl'} isn't GenEval's prompt {i}")
    extra = [d.name for d in images.iterdir() if d.is_dir() and d.name.isdigit() and int(d.name) >= len(rows)]
    if extra:
        raise MeasureError("EVAL", f"{images} holds prompt folders beyond the {len(rows)} scored: {', '.join(sorted(extra)[:3])}")


def manifest(images: Path, rows: list[dict], n_images: int) -> list[dict]:
    """Every image scored, with its sha256: what the record's images_sha256 pins."""
    return [{"file": f"{i:05d}/samples/{k:04d}.png", "sha256": record.sha256_file(sample_path(images, i, k))}
            for i in range(len(rows)) for k in range(n_images)]


# --- scores ---------------------------------------------------------------------------------------


def summarize(rows: list[dict]) -> dict:
    """GenEval's summary_scores.py, computed here too: each task's share of correct images, and their mean."""
    by_tag: dict[str, list[bool]] = defaultdict(list)
    by_prompt: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(bool(r["correct"]))
        by_prompt[r["metadata"]].append(bool(r["correct"]))
    unknown = set(by_tag) - set(TAGS)
    if unknown:
        raise MeasureError("EVAL", f"GenEval's results have tags it doesn't define: {', '.join(sorted(unknown))}")
    tasks = {t: sum(by_tag[t]) / len(by_tag[t]) for t in TAGS if by_tag[t]}
    return {
        "overall": sum(tasks.values()) / len(tasks),
        "tasks": tasks,
        "images_correct": sum(sum(v) for v in by_tag.values()) / len(rows),
        "prompts_correct": sum(any(v) for v in by_prompt.values()) / len(by_prompt),
    }


def read_results(path: Path, images: Path, rows: list[dict], n_images: int) -> list[dict]:
    results = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = {str(sample_path(images, i, k)) for i in range(len(rows)) for k in range(n_images)}
    got = [str(Path(r["filename"])) for r in results]
    if len(got) != len(expected):
        raise MeasureError("EVAL", f"GenEval scored {len(got)} images, expected {len(expected)}, one per image made")
    if set(got) != expected:
        raise MeasureError("EVAL", f"GenEval's results don't name every image made exactly once, e.g. {sorted(expected - set(got))[:1]}")
    return results


def overall_of_summary(text: str) -> float:
    found = re.search(r"Overall score \(avg\. over tasks\): ([0-9.]+)", text)
    if not found:
        raise MeasureError("EVAL", "summary_scores.py printed no overall score")
    return float(found.group(1))


def summary_overall(base: Path, results: Path, log_file: Path) -> float:
    """The overall score GenEval's own summary_scores.py prints for these results."""
    with log_file.open("a") as f:
        proc = subprocess.run(in_evaluator([str(base / "evaluation" / "summary_scores.py"), str(results)]),
                              stdout=subprocess.PIPE, stderr=f, text=True)
        f.write(proc.stdout)
    if proc.returncode != 0:
        raise MeasureError("EVAL", f"GenEval's summary_scores.py exited with code {proc.returncode}: see {log_file}")
    return overall_of_summary(proc.stdout)


def score(base: Path, images: Path, results: Path, detector_dir: Path, info: dict, gpu: int, log_file: Path, log: Log) -> None:
    free, used = free_vram_mib(gpu)
    log(f"gpu memory: {free:.0f} MiB free ({used:.0f} MiB in use by other programs); GenEval's evaluator needs about {SCORE_VRAM_MIB} MiB")
    if free < SCORE_VRAM_MIB + protocol.EVAL_VRAM_MARGIN_MIB:
        raise MeasureError("VRAM_LOW", f"Only {free:.0f} MiB of GPU memory is free and GenEval's evaluator needs about "
                                       f"{SCORE_VRAM_MIB + protocol.EVAL_VRAM_MARGIN_MIB} MiB with the margin")
    cmd = in_evaluator([str(base / "evaluation" / "evaluate_images.py"), str(images), "--outfile", str(results),
                        "--model-path", str(detector_dir), "--model-config", info["config"]], Path(info["preload"]))
    log(f"scoring with GenEval {COMMIT[:7]}'s evaluate_images.py (its output: {log_file})")
    tmp = results.with_name(f".{results.name}.tmp")
    cmd[cmd.index(str(results))] = str(tmp)
    with log_file.open("a") as f:
        f.write("$ " + shlex.join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = [l for l in log_file.read_text(errors="replace").splitlines() if l.strip()][-5:]
        raise MeasureError("EVAL", f"GenEval's evaluate_images.py exited with code {rc}: {' | '.join(tail)}")
    tmp.replace(results)


# --- GPU ------------------------------------------------------------------------------------------

def free_vram_mib(index: int) -> tuple[float, float]:
    gpu = Gpu(index)
    try:
        used, total = gpu.used_bytes(), gpu.total_bytes()
    finally:
        gpu.close()
    return (total - used) / MIB, used / MIB


# --- the command ----------------------------------------------------------------------------------


def main(argv: list[str], log: Log) -> int:
    args = parse_args(argv)
    machine.pin_gpu(args.gpu)
    model_dir = args.model_dir.resolve()
    if not (model_dir / "adapter.py").is_file():
        raise MeasureError("USAGE", f"No adapter.py in {model_dir}")
    model_id = args.model_id or model_dir.name
    out = model_dir / "evals" / args.size / f"{args.benchmark}.json"
    log_path = out.with_suffix(".log") if not args.limit else out.with_name(f"{args.benchmark}.limit.log")
    try:
        return evaluate(args, model_dir, model_id, out, log_path, log)
    except MeasureError as e:
        log(f"FAILED: {e}")
        return finish(log, log_path, e.code, str(e))


def evaluate(args, model_dir: Path, model_id: str, out: Path, log_path: Path, log: Log) -> int:
    bench = protocol.GEN_BENCHMARKS[args.benchmark]
    seeds = list(bench["seeds"])
    if len(seeds) != bench["images_per_prompt"]:
        raise MeasureError("FAILED", "protocol.GEN_BENCHMARKS gives a seed per image")
    log(f"mtr-measure eval {model_id} / {args.size} on GenEval {COMMIT[:7]}" + (f" (first {args.limit} prompts only, no record)" if args.limit else ""))
    log(f"command: {shlex.join(sys.argv)}", echo=False)
    a = ad.load_adapter(model_dir / "adapter.py")
    if getattr(a, "KIND", "image") != "generation":
        raise MeasureError("USAGE", "GenEval scores text-to-image models (KIND = \"generation\")")
    ad.check_size(a, args.size)
    generation.settings(a, args.size)
    precision = ad.choose_precision(list(a.PRECISIONS))

    base = ensure_files(log)
    rows = prompts(base)
    if args.limit:
        rows = rows[: args.limit]
    prompts_sha = FILES["prompts/evaluation_metadata.jsonl"][1]
    images = image_dir(out, args.limit)
    images.mkdir(parents=True, exist_ok=True)
    work_log = images.with_name(f"{images.name}.evaluator.log")
    info = evaluator_info(work_log)
    detector_dir = ensure_weights(log)
    log(f"GenEval's environment: Python {info['python']}, torch {info['torch']} (CUDA {info['cuda']}), mmcv-full {info['mmcv-full']}, "
        f"mmdet {info['mmdet']}, open-clip-torch {info['open-clip-torch']}")

    weights = record.weight_entries([Path(p) for p in a.weight_files(args.size)])
    want = settings_of(a, model_dir, args.size, precision, seeds, prompts_sha, weights)
    state = read_state(images)
    if state is not None and state.get("settings") != want:
        changed = sorted(k for k in want if (state.get("settings") or {}).get(k) != want[k])
        raise MeasureError("USAGE", f"The images in {images} were made with other settings ({', '.join(changed)}). "
                                    f"Delete that directory to make them again")
    state = state or {"settings": want, "generation_s": 0}
    write_state(images, state)
    g = generation.settings(a, args.size)
    log(f"package: {ad.package(a)}, precision {precision}, {g['resolution']}, {g['steps']} steps, guidance {g['guidance']:g}, "
        f"seeds {', '.join(map(str, seeds))} on {g['generator']}; {len(rows)} prompts x {len(seeds)} images")
    make_images(a, model_dir, args.size, precision, rows, seeds, images, state, args.gpu, log, args.pause)
    check_layout(images, rows, len(seeds))

    results = out.with_name(f"{args.benchmark}{'.limit' if args.limit else ''}.results.jsonl")
    t0 = time.perf_counter()
    score(base, images, results, detector_dir, info, args.gpu, work_log, log)
    score_s = time.perf_counter() - t0
    scored = read_results(results, images, rows, len(seeds))
    s = summarize(scored)
    theirs = summary_overall(base, results, work_log)
    if abs(theirs - round(s["overall"], 5)) > 1e-5:
        raise MeasureError("EVAL", f"summary_scores.py says {theirs}, the tool computed {s['overall']:.5f} from the same results")
    for t, v in s["tasks"].items():
        log(f"  {t}: {v:.4f}")
    value = round(s["overall"], 4)
    log(f"GenEval overall: {value} on {len(scored)} images of {len(rows)} prompts (summary_scores.py: {theirs})")
    metrics = {**{t: round(v, 4) for t, v in s["tasks"].items()},
               "images_correct": round(s["images_correct"], 4), "prompts_correct": round(s["prompts_correct"], 4)}
    numbers = {"value": value, "metrics": metrics, "images": len(scored), "prompts": len(rows)}
    if args.limit:
        return finish(log, log_path, "OK", f"GenEval {value} on the first {len(rows)} prompts only: no record written", numbers)

    listing = images.with_name(f"{images.name}.images.json")
    listing.write_text(json.dumps(manifest(images, rows, len(seeds)), indent=1) + "\n")
    gpu = Gpu(args.gpu)
    try:
        gpu_name, driver = gpu.name(), gpu.driver()
    finally:
        gpu.close()
    rec = {
        "kind": "generation",
        "model": model_id,
        "size": args.size,
        "weights": weights,
        "benchmark": {"id": args.benchmark, "name": bench["name"], "split": f"{len(rows)} prompts, {len(seeds)} images each",
                      "n": len(scored), "metric": bench["metric"], "data_sha256": prompts_sha},
        "value": value,
        "metrics": metrics,
        "settings": {
            "precision": precision,
            "resolution": g["resolution"],
            "steps": g["steps"],
            "guidance": float(g["guidance"]),
            "generator": g["generator"],
            "scheduler": state.get("scheduler"),
            "seeds": seeds,
            "seeding": SEEDING,
            "images_per_prompt": len(seeds),
            "prompt_rewriter": "none: GenEval's prompts as written",
            "negative_prompt": "none",
            "image_format": "PNG",
            "evaluator": f"GenEval {COMMIT[:7]}",
            "evaluator_sha256": FILES["evaluation/evaluate_images.py"][1],
            "evaluator_options": OPTIONS,
            "detector": f"{DETECTOR_NAME} (MMDetection), sha256 {DETECTOR.sha256}",
            "color_classifier": f"CLIP ViT-L/14 (OpenAI) through open_clip, sha256 {CLIP.sha256}",
            "evaluator_env": {k: v for k, v in info.items() if k not in ("config", "preload")},
            "images_sha256": record.sha256_file(listing),
            "generation_s": state.get("generation_s"),
            "scoring_s": round(score_s),
        },
        "env": {
            "gpu": gpu_name, "driver": driver, **torch_env(),
            "python": platform.python_version(), "package": ad.package(a),
        },
        "protocol": protocol.PROTOCOL_VERSION,
        "evaluated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_s": round((state.get("generation_s") or 0) + score_s),
        "predictions_sha256": record.sha256_file(results),
        "provenance": machine.provenance(model_dir, None),
    }
    problems = record.validate(rec, "eval-record")
    for prob in problems:
        log(f"SCHEMA: {prob}")
    if problems:
        return finish(log, log_path, "SCHEMA", f"The eval record fails the schema: {problems[0]}", numbers)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    log(f"wrote {out} (the images, {images.name}/, and GenEval's results, {results.name}, stay here as evidence and are never pushed)")
    return finish(log, log_path, "OK", f"{model_id} / {args.size}: GenEval {value}", {"record": str(out), **numbers})
