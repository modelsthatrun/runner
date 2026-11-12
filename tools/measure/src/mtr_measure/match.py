"""Does the CPU compute what the GPU does? (docs/protocol.md, "CPU rows").

A CPU row is not scored on a benchmark. Instead `mtr check` compares its output on the task's timing input with the
GPU's: the default runtime's check saves its prediction as .check/<size>.reference.npz, a CPU runtime's check saves
its own next to it, and compare() reads both. Nothing here needs the GPU, so a CPU check can run while the GPU is busy.

The tolerances are fixed per runtime. pytorch-cpu runs the same code in fp32 instead of fp16, so its outputs should
nearly coincide. onnxruntime-cpu runs the authors' ONNX inference code, whose pre-processing may differ from the
PyTorch code's (a letterbox instead of a stretch), so it gets looser ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import predictions as P

# Saved items: everything at least this far below the task's display threshold, so an object near the threshold on
# one side still finds its match on the other.
SCORE_MARGIN = 0.2

TOLERANCES: dict[str, dict[str, dict[str, float]]] = {
    "pytorch-cpu": {
        "boxes": {"recall": 0.9, "precision": 0.9, "mean_iou": 0.9, "median_score_diff": 0.05},
        "masks": {"recall": 0.9, "precision": 0.9, "mean_iou": 0.85, "median_score_diff": 0.05},
        "keypoints": {"recall": 0.9, "precision": 0.9, "mean_point_error": 0.02, "median_score_diff": 0.05},
        "alpha": {"mean_abs_diff": 0.01, "iou_at_half": 0.98},
        "depth": {"pearson": 0.995, "median_rel_error": 0.02},
    },
    "onnxruntime-cpu": {
        "boxes": {"recall": 0.75, "precision": 0.75, "mean_iou": 0.7, "median_score_diff": 0.15},
        "masks": {"recall": 0.75, "precision": 0.75, "mean_iou": 0.7, "median_score_diff": 0.15},
        "keypoints": {"recall": 0.75, "precision": 0.75, "mean_point_error": 0.05, "median_score_diff": 0.15},
        "alpha": {"mean_abs_diff": 0.03, "iou_at_half": 0.95},
        "depth": {"pearson": 0.98, "median_rel_error": 0.05},
    },
}
# Metrics where a smaller value is better; for the others, larger is.
LOWER_IS_BETTER = {"median_score_diff", "mean_point_error", "mean_abs_diff", "median_rel_error"}


class MatchError(Exception):
    pass


def type_name(pred: Any) -> str:
    for name, cls in (("boxes", P.Boxes), ("masks", P.Masks), ("keypoints", P.Keypoints), ("alpha", P.Alpha), ("depth", P.Depth)):
        if isinstance(pred, cls):
            return name
    raise MatchError(f"No comparison for {type(pred).__name__} outputs yet")


def save(pred: Any, task: str, path: Path) -> None:
    """A prediction on the timing input, as compare() reads it. Scored types keep what clears the threshold's margin."""
    kind = type_name(pred)
    floor = max(0.0, P.THRESHOLDS.get(task, 0.5) - SCORE_MARGIN)
    scores = getattr(pred, "scores", None)
    if scores is not None:
        pred = pred.keep(np.asarray(scores) >= floor)
    arrays: dict[str, Any] = {"type": np.array(kind), "task": np.array(task)}
    if kind == "boxes":
        arrays.update(xyxy=pred.xyxy, scores=pred.scores, labels=np.array(pred.labels, dtype=str))
    elif kind == "masks":
        arrays.update(masks=pred.masks, labels=np.array(pred.labels, dtype=str))
        if pred.scores is not None:
            arrays["scores"] = pred.scores
    elif kind == "keypoints":
        arrays.update(points=pred.points, scores=pred.scores)
        if pred.visible is not None:
            arrays["visible"] = pred.visible
    elif kind == "alpha":
        arrays["alpha"] = pred.alpha
    elif kind == "depth":
        arrays.update(depth=pred.depth, inverse=np.array(pred.inverse))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.savez_compressed(f, **arrays)


def load(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    if a.shape[1:] != b.shape[1:]:
        raise MatchError(f"The masks have different sizes: {list(a.shape[1:])} and {list(b.shape[1:])}")
    fa = a.reshape(len(a), -1).astype(np.float32)
    fb = b.reshape(len(b), -1).astype(np.float32)
    inter = fa @ fb.T
    union = fa.sum(1)[:, None] + fb.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _greedy(iou: np.ndarray, same: np.ndarray, min_iou: float) -> list[tuple[int, int, float]]:
    """Pairs (i, j, iou) matched greedily by IoU, best first, among pairs with the same label and at least min_iou."""
    pairs = [(iou[i, j], i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1]) if same[i, j] and iou[i, j] >= min_iou]
    pairs.sort(reverse=True)
    used_i, used_j, out = set(), set(), []
    for v, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j, float(v)))
    return out


def _instances(ref: dict, got: dict, iou: np.ndarray, threshold: float) -> dict[str, float]:
    """Recall: the reference's items above the threshold that the CPU found; precision: the other way round. An item
    matches one of the other side's items above threshold - SCORE_MARGIN, with the same label."""
    rl = ref.get("labels", np.array([""] * iou.shape[0]))
    gl = got.get("labels", np.array([""] * iou.shape[1]))
    rs = ref.get("scores", np.ones(iou.shape[0]))
    gs = got.get("scores", np.ones(iou.shape[1]))
    same = rl[:, None] == gl[None, :]
    pairs = _greedy(iou, same, 0.5)
    by_ref = {i: (j, v) for i, j, v in pairs}
    by_got = {j: i for i, j, _ in pairs}
    ref_hi = [i for i in range(len(rs)) if rs[i] >= threshold]
    got_hi = [j for j in range(len(gs)) if gs[j] >= threshold]
    recall = sum(1 for i in ref_hi if i in by_ref) / len(ref_hi) if ref_hi else 1.0
    precision = sum(1 for j in got_hi if j in by_got) / len(got_hi) if got_hi else 1.0
    hi_pairs = [(i, j, v) for i, j, v in pairs if rs[i] >= threshold or gs[j] >= threshold]
    return {
        "reference_items": float(len(ref_hi)),
        "cpu_items": float(len(got_hi)),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "mean_iou": round(float(np.mean([v for _, _, v in hi_pairs])) if hi_pairs else 1.0, 4),
        "median_score_diff": round(float(np.median([abs(float(rs[i]) - float(gs[j])) for i, j, _ in hi_pairs])) if hi_pairs else 0.0, 4),
    }


def metrics(ref: dict, got: dict) -> tuple[str, dict[str, float]]:
    kind = str(ref["type"])
    if str(got["type"]) != kind:
        raise MatchError(f"The reference holds {kind} and the CPU output {got['type']}")
    threshold = P.THRESHOLDS.get(str(ref["task"]), 0.5)
    if kind == "boxes":
        return kind, _instances(ref, got, _box_iou(ref["xyxy"], got["xyxy"]), threshold)
    if kind == "masks":
        return kind, _instances(ref, got, _mask_iou(ref["masks"], got["masks"]), threshold)
    if kind == "keypoints":
        rp, gp = ref["points"], got["points"]
        box = lambda p: np.concatenate([p.min(1), p.max(1)], axis=1) if len(p) else np.zeros((0, 4))  # noqa: E731
        m = _instances(ref, got, _box_iou(box(rp), box(gp)), threshold)
        same = np.ones((len(rp), len(gp)), dtype=bool)
        pairs = _greedy(_box_iou(box(rp), box(gp)), same, 0.5)
        errors = []
        for i, j, _ in pairs:
            diag = float(np.hypot(*(rp[i].max(0) - rp[i].min(0)))) or 1.0
            vis = np.ones(rp.shape[1], dtype=bool)
            for side, k in (("visible" in ref and ref, i), ("visible" in got and got, j)):
                if side:
                    vis &= side["visible"][k]
            if vis.any():
                errors.append(float(np.mean(np.hypot(*(rp[i][vis] - gp[j][vis]).T))) / diag)
        m.pop("mean_iou")
        m["mean_point_error"] = round(float(np.mean(errors)), 4) if errors else 0.0
        return kind, m
    if kind == "alpha":
        a, b = ref["alpha"].astype(np.float32), got["alpha"].astype(np.float32)
        if a.shape != b.shape:
            raise MatchError(f"The mattes have different sizes: {list(a.shape)} and {list(b.shape)}")
        fa, fb = a >= 0.5, b >= 0.5
        union = float((fa | fb).sum())
        return kind, {"mean_abs_diff": round(float(np.abs(a - b).mean()), 5),
                      "iou_at_half": round(float((fa & fb).sum()) / union, 5) if union else 1.0}
    if kind == "depth":
        a, b = ref["depth"].astype(np.float64), got["depth"].astype(np.float64)
        if a.shape != b.shape:
            raise MatchError(f"The depth maps have different sizes: {list(a.shape)} and {list(b.shape)}")
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 2:
            raise MatchError("The depth maps have no finite values to compare")
        a, b = a[ok], b[ok]
        pearson = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 0.0
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-6)
        return kind, {"pearson": round(pearson, 5), "median_rel_error": round(float(np.median(rel)), 5)}
    raise MatchError(f"No comparison for {kind} outputs yet")


def compare(reference: Path, output: Path, runtime: str) -> dict:
    """{"type", "reference", "metrics", "tolerance", "passed", "failed": [metric, ...]} for the record's runtime.match."""
    kind, m = metrics(load(reference), load(output))
    tol = TOLERANCES[runtime][kind]
    failed = [k for k, limit in tol.items() if (m[k] > limit if k in LOWER_IS_BETTER else m[k] < limit)]
    return {"type": kind, "reference": "pytorch-cuda", "metrics": m, "tolerance": tol, "passed": not failed, "failed": failed}


def describe(result: dict) -> str:
    parts = []
    for k, limit in result["tolerance"].items():
        sign = "≤" if k in LOWER_IS_BETTER else "≥"
        parts.append(f"{k} {result['metrics'][k]:g} (needs {sign} {limit:g})")
    return ", ".join(parts)
