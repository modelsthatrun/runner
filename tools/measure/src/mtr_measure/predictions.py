"""Standard prediction types for the output step (docs/adapters.md).

An adapter's to_prediction(model, x, out) turns predict()'s output into one of these. The shared
drawing code (render.py) then draws every model of a task the same way, so outputs stay comparable.
Measuring never uses them.

Every type is in pixels of the input image as its file decodes, the same image the default
prepare_input() gives predict(). Labels are class names, not ids, so a class gets the same color in
every model. Return every score: the tool applies the task's fixed threshold (THRESHOLDS) itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


class PredictionError(Exception):
    pass


def _scores(value: Any, n: int, what: str) -> np.ndarray:
    s = np.asarray(value, dtype=np.float32).reshape(-1)
    if s.shape != (n,):
        raise PredictionError(f"{what}: {n} items but {s.shape[0]} scores")
    if not np.all(np.isfinite(s)) or s.min(initial=0) < 0 or s.max(initial=0) > 1:
        raise PredictionError(f"{what}: scores must be probabilities in [0, 1], not logits")
    return s


def _labels(value: Sequence[Any], n: int, what: str) -> list[str]:
    labels = [str(v) for v in value]
    if len(labels) != n:
        raise PredictionError(f"{what}: {n} items but {len(labels)} labels")
    return labels


@dataclass
class Boxes:
    """Object detection. xyxy is (N, 4): x1, y1, x2, y2."""

    xyxy: np.ndarray
    scores: np.ndarray
    labels: list[str]

    def __post_init__(self) -> None:
        b = np.asarray(self.xyxy, dtype=np.float32)
        if b.size == 0:
            b = b.reshape(0, 4)
        if b.ndim != 2 or b.shape[1] != 4:
            raise PredictionError(f"Boxes: xyxy must be (N, 4), got {list(b.shape)}")
        if not np.all(np.isfinite(b)):
            raise PredictionError("Boxes: xyxy has NaN or infinite values")
        if np.any(b[:, 2] < b[:, 0]) or np.any(b[:, 3] < b[:, 1]):
            raise PredictionError("Boxes: some boxes have x2 < x1 or y2 < y1. Are they xywh or cxcywh instead of xyxy?")
        self.xyxy = b
        self.scores = _scores(self.scores, len(b), "Boxes")
        self.labels = _labels(self.labels, len(b), "Boxes")

    def __len__(self) -> int:
        return len(self.xyxy)

    def keep(self, which: np.ndarray) -> Boxes:
        return Boxes(self.xyxy[which], self.scores[which], [l for l, k in zip(self.labels, which) if k])


@dataclass
class Masks:
    """Instance or semantic segmentation. masks is (N, H, W), one boolean mask per instance or class,
    at the input image's size. Semantic models have no scores: build them with from_class_map()."""

    masks: np.ndarray
    labels: list[str]
    scores: np.ndarray | None = None

    def __post_init__(self) -> None:
        m = np.asarray(self.masks)
        if m.ndim != 3:
            raise PredictionError(f"Masks: masks must be (N, H, W), got {list(m.shape)}")
        if m.dtype != np.bool_:
            if np.issubdtype(m.dtype, np.floating):
                raise PredictionError("Masks: masks are floats. Threshold them to booleans in to_prediction()")
            m = m != 0
        self.masks = m
        self.labels = _labels(self.labels, len(m), "Masks")
        if self.scores is not None:
            self.scores = _scores(self.scores, len(m), "Masks")

    @classmethod
    def from_class_map(cls, class_map: Any, names: Sequence[str], background: int | None = None) -> Masks:
        """One mask per class present in an (H, W) map of class indices. `background` is left out."""
        cm = np.asarray(class_map)
        if cm.ndim != 2 or not np.issubdtype(cm.dtype, np.integer):
            raise PredictionError(f"Masks: a class map must be (H, W) integers, got {list(cm.shape)} {cm.dtype}")
        present = [int(i) for i in np.unique(cm) if i != background]
        if present and (present[0] < 0 or present[-1] >= len(names)):
            raise PredictionError(f"Masks: class index {present[-1]} is outside the {len(names)} class names")
        masks = np.stack([cm == i for i in present]) if present else np.zeros((0, *cm.shape), dtype=bool)
        return cls(masks, [names[i] for i in present])

    def __len__(self) -> int:
        return len(self.masks)

    def keep(self, which: np.ndarray) -> Masks:
        scores = None if self.scores is None else self.scores[which]
        return Masks(self.masks[which], [l for l, k in zip(self.labels, which) if k], scores)


# The four types below are defined so adapters can return them, but render.py doesn't draw them
# yet. Their fields may change when their drawing code is built.


@dataclass
class Alpha:
    """Image matting. alpha is (H, W) in [0, 1], 1 for the foreground."""

    alpha: np.ndarray

    def __post_init__(self) -> None:
        self.alpha = np.asarray(self.alpha, dtype=np.float32)
        if self.alpha.ndim != 2:
            raise PredictionError(f"Alpha: alpha must be (H, W), got {list(self.alpha.shape)}")


@dataclass
class Depth:
    """Depth estimation. depth is (H, W), relative or metric. inverse is True for disparity, where
    larger values are nearer."""

    depth: np.ndarray
    inverse: bool = False

    def __post_init__(self) -> None:
        self.depth = np.asarray(self.depth, dtype=np.float32)
        if self.depth.ndim != 2:
            raise PredictionError(f"Depth: depth must be (H, W), got {list(self.depth.shape)}")


@dataclass
class Keypoints:
    """Pose estimation. points is (N, K, 2), one row of K keypoints per instance. names holds the K
    keypoint names, and skeleton the pairs of keypoint indices to connect. visible is (N, K) booleans
    for models whose official code hides some keypoints when it draws them; None shows them all."""

    points: np.ndarray
    scores: np.ndarray
    names: list[str]
    skeleton: list[tuple[int, int]] = field(default_factory=list)
    visible: np.ndarray | None = None

    def __post_init__(self) -> None:
        p = np.asarray(self.points, dtype=np.float32)
        if p.ndim != 3 or p.shape[2] != 2:
            raise PredictionError(f"Keypoints: points must be (N, K, 2), got {list(p.shape)}")
        if len(self.names) != p.shape[1]:
            raise PredictionError(f"Keypoints: {p.shape[1]} keypoints per instance but {len(self.names)} names")
        self.points = p
        self.scores = _scores(self.scores, len(p), "Keypoints")
        if self.visible is not None:
            v = np.asarray(self.visible, dtype=bool)
            if v.shape != p.shape[:2]:
                raise PredictionError(f"Keypoints: visible must be (N, K) = {list(p.shape[:2])}, got {list(v.shape)}")
            self.visible = v

    def __len__(self) -> int:
        return len(self.points)

    def keep(self, which: np.ndarray) -> Keypoints:
        visible = None if self.visible is None else self.visible[which]
        return replace(self, points=self.points[which], scores=self.scores[which], visible=visible)


@dataclass
class Text:
    """OCR. polygons is (N, P, 2), one outline per text region, and texts what was read in each."""

    polygons: np.ndarray
    texts: list[str]
    scores: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.polygons, dtype=np.float32)
        if p.ndim != 3 or p.shape[2] != 2:
            raise PredictionError(f"Text: polygons must be (N, P, 2), got {list(p.shape)}")
        self.polygons = p
        self.texts = _labels(self.texts, len(p), "Text")
        self.scores = _scores(self.scores, len(p), "Text")

    def __len__(self) -> int:
        return len(self.polygons)

    def keep(self, which: np.ndarray) -> Text:
        return Text(self.polygons[which], [t for t, k in zip(self.texts, which) if k], self.scores[which])


TYPES = (Boxes, Masks, Alpha, Depth, Keypoints, Text)

# The prediction type each task's outputs use. semantic-segmentation isn't a catalog task yet; the
# DeepLabV3 example in tools/measure/examples/ uses it.
TASK_TYPES: dict[str, type] = {
    "object-detection": Boxes,
    "instance-segmentation": Masks,
    "semantic-segmentation": Masks,
    "image-matting": Alpha,
    "depth-estimation": Depth,
    "pose-estimation": Keypoints,
    "ocr": Text,
}

# Names some models use for COCO classes (VOC-style or older label files). The output step and the
# COCO evaluation both use the COCO name, so a class is drawn with the same name and color in every model.
LABEL_ALIASES: dict[str, str] = {
    "motorbike": "motorcycle", "aeroplane": "airplane", "sofa": "couch", "tvmonitor": "tv",
    "diningtable": "dining table", "pottedplant": "potted plant", "hand bag": "handbag",
}


def canonical_labels(pred: Any) -> Any:
    """The prediction with LABEL_ALIASES applied to its labels. Types without labels pass."""
    labels = getattr(pred, "labels", None)
    if not labels or not any(l in LABEL_ALIASES for l in labels):
        return pred
    return replace(pred, labels=[LABEL_ALIASES.get(l, l) for l in labels])


# The fixed score threshold per task, the same for every model. docs/protocol.md ("Fixed
# thresholds") is where it is decided; keep this in step with it (a test checks that they agree).
THRESHOLDS: dict[str, float] = {
    "object-detection": 0.5,
    "instance-segmentation": 0.5,
    "pose-estimation": 0.5,
}


def task_type(task: str) -> type:
    if task not in TASK_TYPES:
        raise PredictionError(f'Unknown task "{task}". The output step knows {", ".join(TASK_TYPES)}')
    return TASK_TYPES[task]


def apply_threshold(pred: Any, task: str) -> Any:
    """Keep the predictions whose score is at least the task's threshold. Types without scores pass."""
    scores = getattr(pred, "scores", None)
    if scores is None:
        return pred
    threshold = THRESHOLDS.get(task)
    if threshold is None:
        raise PredictionError(
            f'The predictions have scores, but "{task}" has no fixed threshold yet. '
            "It is decided with the catalog's maintainers in docs/protocol.md (\"Fixed thresholds\"), then added to THRESHOLDS"
        )
    return pred.keep(scores >= threshold)
