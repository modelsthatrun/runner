"""The shared drawing code of the output step: one look per prediction type, the same for every model.

Each output is drawn on the input image at its own size, never cropped or resized, so pages can lay
it over the test image (the input/output slider and the loupe). Strokes and text scale with the
image. The look follows the site's illustrations: a solid box or mask
per object in its class color, with a filled tag holding the class name and score; the alpha matte in
white on black; depth in the turbo colormap, near in red; and the skeleton in pink on a dimmed photo.

Text (OCR) isn't built yet and raises NotImplementedError.
"""

from __future__ import annotations

import io
import zlib
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .predictions import Alpha, Boxes, Depth, Keypoints, Masks, PredictionError, Text

WEBP_QUALITY = 88  # the same quality the catalog uses when it converts other formats

# Class colors. Common classes keep the colors the site's fixtures use. Any other class name gets
# one from PALETTE by a stable hash, so a class looks the same in every model and every run.
# PALETTE leaves out the common classes' colors, so no other class is drawn like a person. Two
# other classes can still share a color; their tags tell them apart (docs/ux.md, principle 6).
# Changing either means rendering every model's outputs again.
CLASS_COLORS = {
    "person": "#4d9dff", "car": "#ffb020", "dog": "#2ee6a8", "cup": "#ff7a59",
    "laptop": "#a78bfa", "potted plant": "#7ddc5a", "book": "#ffd84d",
    # Not in the fixtures: the hash gives couch the same color as cat, and cats lie on couches.
    "couch": "#9aa7b8",
}
PALETTE = ["#3fd7e8", "#ff5fa2", "#d17bff", "#ff4f4f", "#f2f2f2", "#c9a27e", "#b8e62e", "#14b8a6"]
TAG_TEXT = (11, 13, 16)  # #0b0d10, dark text on the colored tag
DIM = (5, 7, 10)  # #05070a, laid over the photo under masks
DIM_OPACITY = 0.45
MASK_OPACITY = 0.6
POSE_DIM_OPACITY = 0.5
POSE_COLOR = (0xFF, 0x5F, 0xA2)  # #ff5fa2, the fixtures' skeleton
# The fixtures' turbo colormap (scenes.ts), from far (0) to near (1).
TURBO = ["#30123b", "#4145ab", "#4675ed", "#39a2fc", "#1bcfd4", "#24eca6", "#61fc6c", "#a4fc3b", "#d1e834",
         "#f3c63a", "#fe9b2d", "#f36315", "#d93806", "#b11901", "#7a0403"]
# Depth is scaled per image between these percentiles of its nearness, so a few stray pixels don't
# flatten the rest. Depth maps are relative to their own image, never compared in absolute units.
DEPTH_PERCENTILES = (2, 98)


def hex_rgb(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def class_color(label: str) -> tuple[int, int, int]:
    return hex_rgb(CLASS_COLORS.get(label) or PALETTE[zlib.crc32(label.encode()) % len(PALETTE)])


def stroke_width(image: Image.Image) -> int:
    return max(2, round(max(image.size) / 300))


def font_for(image: Image.Image) -> ImageFont.FreeTypeFont:
    font = ImageFont.load_default(size=max(12, round(max(image.size) / 60)))
    if not isinstance(font, ImageFont.FreeTypeFont):
        raise RuntimeError("Pillow was built without FreeType, so tag text can't be scaled")
    return font


# How many tag heights a tag may move down to clear earlier tags. In a crowd it then stays by its
# object and may overlap, rather than sliding down the image away from it.
TAG_STEPS = 3


def draw_tags(image: Image.Image, tags: list[tuple[float, float, str, tuple[int, int, int]]]) -> None:
    """Filled tags with text. Each sits on top of its anchor point, or below it at the top edge, and
    moves down a little until it no longer covers an earlier tag."""
    d = ImageDraw.Draw(image)
    font = font_for(image)
    ascent, descent = font.getmetrics()
    pad = max(2, font.size // 4)
    h = ascent + descent + 2 * pad
    placed: list[tuple[int, int, int, int]] = []
    for x, y, text, color in tags:
        w = round(font.getlength(text)) + 2 * pad
        left = int(min(max(x, 0), max(image.width - w, 0)))
        first = top = int(y - h if y - h >= 0 else max(y, 0))
        covers = lambda top: any(left < r and l < left + w and top < b and t < top + h for l, t, r, b in placed)
        for _ in range(TAG_STEPS):
            if not covers(top) or top + 2 * h > image.height:
                break
            top += h
        if covers(top):
            top = first
        placed.append((left, top, left + w, top + h))
        d.rectangle([left, top, left + w - 1, top + h - 1], fill=color)
        d.text((left + pad, top + pad + ascent), text, font=font, fill=TAG_TEXT, anchor="ls")


def tag_text(label: str, score: float | None) -> str:
    return label if score is None else f"{label} {score:.2f}"


def draw_boxes(image: Image.Image, pred: Boxes) -> Image.Image:
    out = image.convert("RGB")
    d = ImageDraw.Draw(out)
    lw = stroke_width(out)
    w, h = out.size
    order = np.argsort(pred.scores, kind="stable")  # higher scores are drawn last, on top
    tags = []
    for i in order:
        x1, y1, x2, y2 = pred.xyxy[i]
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)
        color = class_color(pred.labels[i])
        d.rectangle([float(x1), float(y1), float(x2), float(y2)], outline=color, width=lw)
        tags.append((float(x1), float(y1), tag_text(pred.labels[i], float(pred.scores[i])), color))
    draw_tags(out, tags)
    return out


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    """Shrink a boolean mask by r pixels (4-neighbourhood). The image border doesn't erode it."""
    e = mask.copy()
    for _ in range(r):
        n = e.copy()
        n[1:, :] &= e[:-1, :]
        n[:-1, :] &= e[1:, :]
        n[:, 1:] &= e[:, :-1]
        n[:, :-1] &= e[:, 1:]
        e = n
    return e


def draw_masks(image: Image.Image, pred: Masks) -> Image.Image:
    base = image.convert("RGB")
    check_size(base, pred.masks.shape[1:], "The masks")
    px = np.asarray(base, dtype=np.float32)
    px = px * (1 - DIM_OPACITY) + np.array(DIM, dtype=np.float32) * DIM_OPACITY
    lw = stroke_width(base)
    areas = pred.masks.sum(axis=(1, 2))
    tags = []
    for i in np.argsort(-areas, kind="stable"):  # large masks first, so small ones stay visible
        m = pred.masks[i]
        if not areas[i]:
            continue
        color = np.array(class_color(pred.labels[i]), dtype=np.float32)
        px[m] = px[m] * (1 - MASK_OPACITY) + color * MASK_OPACITY
        px[m & ~erode(m, lw)] = color
        top = int(np.argmax(m.any(axis=1)))  # the tag goes on the mask's topmost pixel
        left = int(np.argmax(m[top]))
        score = None if pred.scores is None else float(pred.scores[i])
        tags.append((float(left), float(top), tag_text(pred.labels[i], score), tuple(int(c) for c in color)))
    out = Image.fromarray(np.clip(px + 0.5, 0, 255).astype(np.uint8), "RGB")
    draw_tags(out, tags)
    return out


def check_size(image: Image.Image, shape: tuple[int, ...], what: str) -> None:
    w, h = image.size
    if shape != (h, w):
        raise PredictionError(f"{what} is {shape[1]}x{shape[0]} but the image is {w}x{h}. to_prediction() must return it at the input image's size")


def draw_alpha(image: Image.Image, pred: Alpha) -> Image.Image:
    # The matte itself, white for the foreground on black, so soft edges such as hair stay visible.
    check_size(image, pred.alpha.shape, "The alpha matte")
    a = np.nan_to_num(pred.alpha, nan=0.0).clip(0, 1)
    gray = (a * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(gray, "L").convert("RGB")


def turbo(t: np.ndarray) -> np.ndarray:
    """The fixtures' turbo colormap at t in [0, 1], as (..., 3) floats."""
    stops = np.array([hex_rgb(c) for c in TURBO], dtype=np.float32)
    x = t.clip(0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(np.int64), len(stops) - 2)
    f = (x - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def draw_depth(image: Image.Image, pred: Depth) -> Image.Image:
    # Nearness: disparity as it is, metric or relative depth inverted. Pixels without a valid value
    # (NaN, infinite, or depth at or below zero) are drawn as farthest.
    check_size(image, pred.depth.shape, "The depth map")
    d = pred.depth.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        near = d if pred.inverse else np.where(d > 0, 1 / d, np.nan)
    valid = np.isfinite(near)
    t = np.zeros(d.shape, dtype=np.float32)
    if valid.any():
        lo, hi = np.percentile(near[valid], DEPTH_PERCENTILES)
        t[valid] = ((near[valid] - lo) / (hi - lo)).clip(0, 1) if hi > lo else 0.5
    return Image.fromarray((turbo(t) + 0.5).astype(np.uint8), "RGB")


def draw_keypoints(image: Image.Image, pred: Keypoints) -> Image.Image:
    base = image.convert("RGB")
    px = np.asarray(base, dtype=np.float32)
    px = px * (1 - POSE_DIM_OPACITY) + np.array(DIM, dtype=np.float32) * POSE_DIM_OPACITY
    out = Image.fromarray(np.clip(px + 0.5, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(out)
    lw = stroke_width(out)
    r = lw * 1.6
    tags = []
    for i in np.argsort(pred.scores, kind="stable"):  # higher scores are drawn last, on top
        pts = pred.points[i]
        shown = np.isfinite(pts).all(axis=1)
        if pred.visible is not None:
            shown &= pred.visible[i]
        if not shown.any():
            continue
        for a, b in pred.skeleton:
            if shown[a] and shown[b]:
                d.line([tuple(map(float, pts[a])), tuple(map(float, pts[b]))], fill=POSE_COLOR, width=lw * 2)
        for x, y in pts[shown]:
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255), outline=POSE_COLOR, width=lw)
        top = int(np.argmin(np.where(shown, pts[:, 1], np.inf)))  # the tag goes above the topmost keypoint
        tags.append((float(pts[top, 0] - r), float(pts[top, 1] - r), f"person {float(pred.scores[i]):.2f}", POSE_COLOR))
    draw_tags(out, tags)
    return out


# One entry per standard type. None means its drawing code isn't built yet.
DRAWERS: dict[type, Callable[[Image.Image, Any], Image.Image] | None] = {
    Boxes: draw_boxes,
    Masks: draw_masks,
    Alpha: draw_alpha,
    Depth: draw_depth,
    Keypoints: draw_keypoints,
    Text: None,
}


def drawer(kind: type) -> Callable[[Image.Image, Any], Image.Image]:
    if kind not in DRAWERS:
        raise PredictionError(f"{kind.__name__} is not a standard prediction type (mtr_measure/predictions.py)")
    draw = DRAWERS[kind]
    if draw is None:
        raise NotImplementedError(
            f"Drawing {kind.__name__} outputs isn't built yet. Add it to tools/measure/src/mtr_measure/render.py "
            "first, so every model of the task is drawn the same way (docs/adapters.md)"
        )
    return draw


def render(image: Image.Image, pred: Any) -> Image.Image:
    return drawer(type(pred))(image, pred)


def encode_webp(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "WEBP", quality=WEBP_QUALITY, method=6)
    return buf.getvalue()
