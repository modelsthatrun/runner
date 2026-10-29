"""Tests for the output step: prediction types, thresholds, drawing and the CLI. No GPU or torch
needed. Run: uv run --frozen --project tools/measure pytest tools/measure/tests -q"""

import io
import re
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mtr_measure import adapter as ad
from mtr_measure import outputs, predictions, render
from mtr_measure.predictions import Alpha, Boxes, Depth, Keypoints, Masks, PredictionError, Text

REPO = Path(__file__).resolve().parents[3]
GREY = (128, 128, 128)


def boxes(*rows):
    """Boxes from (x1, y1, x2, y2, score, label) rows."""
    return Boxes([r[:4] for r in rows], [r[4] for r in rows], [r[5] for r in rows])


def test_boxes_validate():
    assert len(Boxes(np.zeros((0, 4)), [], [])) == 0
    assert len(Boxes([], [], [])) == 0
    with pytest.raises(PredictionError, match=r"\(N, 4\)"):
        Boxes([[1, 2, 3]], [0.9], ["cat"])
    with pytest.raises(PredictionError, match="xywh"):
        boxes((50, 10, 20, 40, 0.9, "cat"))
    with pytest.raises(PredictionError, match="logits"):
        boxes((0, 0, 10, 10, 3.2, "cat"))
    with pytest.raises(PredictionError, match="labels"):
        Boxes([[0, 0, 10, 10]], [0.9], ["cat", "dog"])
    with pytest.raises(PredictionError, match="NaN"):
        boxes((0, 0, float("nan"), 10, 0.9, "cat"))


def test_masks_from_class_map():
    cm = np.array([[0, 0, 2], [0, 2, 2], [1, 1, 0]], dtype=np.uint8)
    m = Masks.from_class_map(cm, ["background", "cat", "dog"], background=0)
    assert m.labels == ["cat", "dog"] and m.scores is None
    assert m.masks.dtype == np.bool_ and m.masks.shape == (2, 3, 3)
    assert m.masks[1].sum() == 3
    assert Masks.from_class_map(cm, ["wall", "cat", "dog"]).labels == ["wall", "cat", "dog"]
    assert len(Masks.from_class_map(np.zeros((4, 4), np.int64), ["background"], background=0)) == 0
    with pytest.raises(PredictionError, match="outside"):
        Masks.from_class_map(cm, ["background", "cat"])
    with pytest.raises(PredictionError, match="floats"):
        Masks(np.full((1, 3, 3), 0.7), ["cat"])
    assert Masks(np.ones((1, 2, 2), np.uint8), ["cat"]).masks.dtype == np.bool_


def test_threshold_keeps_scores_at_or_above_it():
    b = boxes((0, 0, 9, 9, 0.5, "cat"), (0, 0, 9, 9, 0.49, "dog"), (1, 1, 5, 5, 0.97, "person"))
    kept = predictions.apply_threshold(b, "object-detection")
    assert kept.labels == ["cat", "person"]
    assert kept.scores.tolist() == pytest.approx([0.5, 0.97])
    assert kept.xyxy.shape == (2, 4)

    semantic = Masks(np.ones((1, 2, 2), bool), ["cat"])
    assert predictions.apply_threshold(semantic, "semantic-segmentation") is semantic
    scored = Text(np.zeros((1, 4, 2)), ["OPEN"], [0.9])
    with pytest.raises(PredictionError, match="no fixed threshold"):
        predictions.apply_threshold(scored, "ocr")
    people = Keypoints(np.zeros((2, 17, 2)), [0.9, 0.3], ["nose"] * 17, visible=np.ones((2, 17), bool))
    kept = predictions.apply_threshold(people, "pose-estimation")
    assert len(kept) == 1 and kept.visible.shape == (1, 17)


def test_alias_labels_get_the_coco_name():
    b = predictions.canonical_labels(boxes((0, 0, 9, 9, 0.9, "sofa"), (0, 0, 9, 9, 0.8, "cat")))
    assert b.labels == ["couch", "cat"]
    alpha = Alpha(np.zeros((2, 2)))
    assert predictions.canonical_labels(alpha) is alpha


def test_thresholds_match_the_protocol():
    text = (REPO / "docs" / "protocol.md").read_text()
    table = text.split("### Fixed thresholds", 1)[1].split("\n#", 1)[0]
    rows = re.findall(r"^\|\s*([a-z0-9-]+)\s*\|\s*([0-9.]+)", table, re.M)
    assert rows, "docs/protocol.md has no Fixed thresholds table"
    assert {task: float(v) for task, v in rows} == predictions.THRESHOLDS


def test_task_types():
    assert predictions.task_type("object-detection") is Boxes
    assert predictions.task_type("instance-segmentation") is Masks
    with pytest.raises(PredictionError, match="Unknown task"):
        predictions.task_type("teleportation")


def test_to_prediction_from_the_adapter():
    b = boxes((0, 0, 4, 4, 0.9, "cat"))
    a = types.SimpleNamespace(to_prediction=lambda model, x, out: b)
    assert ad.to_prediction(a, None, None, {"raw": 1}, "object-detection") is b
    # predict() may return a standard type itself
    assert ad.to_prediction(types.SimpleNamespace(), None, None, b, "object-detection") is b
    with pytest.raises(ad.AdapterError, match="Add to_prediction"):
        ad.to_prediction(types.SimpleNamespace(), None, None, {"raw": 1}, "object-detection")
    with pytest.raises(ad.AdapterError, match="are Masks, but to_prediction\\(\\) returned Boxes"):
        ad.to_prediction(a, None, None, None, "instance-segmentation")

    def todo(model, x, out):
        raise NotImplementedError

    with pytest.raises(ad.AdapterError, match="isn't written yet"):
        ad.to_prediction(types.SimpleNamespace(to_prediction=todo), None, None, None, "object-detection")


def test_class_colors_are_stable_and_keep_common_classes_apart():
    assert render.class_color("person") == (0x4D, 0x9D, 0xFF)
    assert render.class_color("zebra") == render.class_color("zebra")
    fixed = {render.hex_rgb(c) for c in render.CLASS_COLORS.values()}
    others = ["sofa", "cat", "remote", "bicycle", "zebra", "traffic light"] + [f"class-{i}" for i in range(200)]
    assert not {render.class_color(label) for label in others} & fixed
    assert render.class_color("cat") != render.class_color("couch")


def test_draw_boxes():
    image = Image.new("RGB", (300, 200), GREY)
    out = render.render(image, boxes((50, 80, 150, 180, 0.9, "person")))
    assert out.size == image.size and image.getpixel((50, 80)) == GREY
    color = render.class_color("person")
    assert out.getpixel((50, 130)) == color  # left edge
    assert out.getpixel((150, 130)) == color  # right edge
    assert out.getpixel((100, 130)) == GREY  # inside the box
    assert out.getpixel((280, 20)) == GREY  # away from it
    # The tag sits on top of the box, in the class color.
    assert out.getpixel((52, 78)) == color


def test_draw_boxes_clips_and_keeps_the_tag_inside():
    image = Image.new("RGB", (120, 90), GREY)
    out = render.render(image, boxes((-10, -5, 200, 60, 0.8, "car")))
    color = render.class_color("car")
    assert out.size == (120, 90)
    assert out.getpixel((0, 30)) == color and out.getpixel((119, 30)) == color
    assert out.getpixel((60, 75)) == GREY


def test_draw_masks():
    image = Image.new("RGB", (200, 120), GREY)
    m = np.zeros((1, 120, 200), bool)
    m[0, 40:100, 50:150] = True
    out = np.asarray(render.render(image, Masks(m, ["cat"])), dtype=int)
    color = np.array(render.class_color("cat"))
    dim = np.array(GREY) * (1 - render.DIM_OPACITY) + np.array(render.DIM) * render.DIM_OPACITY
    tint = dim * (1 - render.MASK_OPACITY) + color * render.MASK_OPACITY
    assert out.shape == (120, 200, 3)
    assert np.abs(out[110, 10] - dim).max() <= 1  # outside: the photo, dimmed
    assert np.abs(out[70, 100] - tint).max() <= 1  # inside: tinted with the class color
    assert (out[99, 100] == color).all() and (out[70, 50] == color).all()  # the outline
    assert (out[38, 51] == color).all()  # the tag, on top of the mask


def test_draw_masks_with_none_left():
    # Every mask can fall under the threshold; the output is then the dimmed photo.
    out = render.render(Image.new("RGB", (20, 10), GREY), Masks(np.zeros((0, 10, 20), bool), [], []))
    assert out.size == (20, 10)


def test_tags_do_not_cover_each_other():
    image = Image.new("RGB", (200, 120), GREY)
    out = render.render(image, boxes((10, 0, 190, 110, 0.9, "person"), (12, 0, 100, 60, 0.8, "car")))
    # Both boxes start at the top edge, so both tags start inside them. The later one moves down.
    column = [out.getpixel((14, y)) for y in range(0, 50)]  # above the car box's bottom edge
    assert render.class_color("person") in column and render.class_color("car") in column


def test_tags_in_a_crowd_stay_by_their_object():
    image = Image.new("RGB", (200, 200), GREY)
    crowd = boxes(*[(10, 20, 60, 40, 0.9 - i / 100, "sheep") for i in range(12)])
    out = render.render(image, crowd)
    # Twelve tags on the same spot don't slide down to the bottom of the image.
    assert render.class_color("sheep") not in [out.getpixel((14, y)) for y in range(120, 200)]


def test_masks_must_match_the_image_size():
    image = Image.new("RGB", (60, 40), GREY)
    with pytest.raises(PredictionError, match="60x40"):
        render.render(image, Masks(np.ones((1, 20, 30), bool), ["cat"]))


def test_keypoints_validate():
    with pytest.raises(PredictionError, match="names"):
        Keypoints(np.zeros((1, 17, 2)), [0.9], ["nose"])
    with pytest.raises(PredictionError, match="visible"):
        Keypoints(np.zeros((1, 2, 2)), [0.9], ["a", "b"], visible=np.ones((1, 3), bool))


def test_draw_alpha():
    a = np.zeros((40, 60), np.float32)
    a[10:30, 20:40] = 1
    a[0, 0] = 0.5
    out = render.render(Image.new("RGB", (60, 40), GREY), Alpha(a))
    assert out.size == (60, 40) and out.mode == "RGB"
    assert out.getpixel((30, 20)) == (255, 255, 255) and out.getpixel((5, 35)) == (0, 0, 0)
    assert out.getpixel((0, 0)) == (128, 128, 128)
    with pytest.raises(PredictionError, match="60x40"):
        render.render(Image.new("RGB", (60, 40)), Alpha(np.zeros((20, 30))))


def test_draw_depth_puts_near_in_red_and_far_in_blue():
    near, far = render.hex_rgb(render.TURBO[-1]), render.hex_rgb(render.TURBO[0])
    meters = np.full((20, 30), 10.0, np.float32)
    meters[:, :10] = 1.0  # the left third is near
    meters[0, 29] = np.inf  # an invalid pixel is drawn as farthest
    out = render.render(Image.new("RGB", (30, 20)), Depth(meters))
    assert out.getpixel((5, 10)) == near and out.getpixel((25, 10)) == far and out.getpixel((29, 0)) == far
    # Disparity is already nearness, so the same picture comes from 1 / depth.
    same = render.render(Image.new("RGB", (30, 20)), Depth(1 / np.where(np.isfinite(meters), meters, 1e9), inverse=True))
    assert same.getpixel((5, 10)) == near and same.getpixel((25, 10)) == far
    flat = render.render(Image.new("RGB", (30, 20)), Depth(np.ones((20, 30))))
    assert flat.getpixel((3, 3)) == tuple(int(c + 0.5) for c in render.turbo(np.array(0.5)))


def test_draw_keypoints():
    image = Image.new("RGB", (200, 120), GREY)
    pts = np.array([[[50, 60], [150, 60], [100, 100]]], np.float32)
    out = render.render(image, Keypoints(pts, [0.9], ["a", "b", "c"], [(0, 1), (1, 2)], visible=[[True, True, False]]))
    dim = np.array(GREY) * (1 - render.POSE_DIM_OPACITY) + np.array(render.DIM) * render.POSE_DIM_OPACITY
    assert np.abs(np.array(out.getpixel((10, 110))) - dim).max() <= 1  # the photo, dimmed
    assert out.getpixel((100, 60)) == render.POSE_COLOR  # the a-b limb
    assert out.getpixel((50, 60)) == (255, 255, 255)  # a keypoint
    assert np.abs(np.array(out.getpixel((100, 100))) - dim).max() <= 1  # c is hidden, and so is b-c


def test_text_is_not_drawn_yet():
    with pytest.raises(NotImplementedError, match="Drawing Text outputs isn't built yet"):
        render.render(Image.new("RGB", (4, 4)), Text(np.zeros((1, 4, 2)), ["OPEN"], [0.9]))


def test_webp_round_trip():
    data = render.encode_webp(Image.new("RGB", (64, 48), GREY))
    with Image.open(io.BytesIO(data)) as im:
        assert im.format == "WEBP" and im.size == (64, 48)


# ── The CLI, with a CPU-only fake adapter

FAKE_ADAPTER = '''
import numpy as np
from mtr_measure.predictions import Boxes

SIZES = ["s"]
PRECISIONS = ["fp32"]
PACKAGE = "pytest"
INPUT = {"s": "64x48"}
FAIL_ON = None

def load(size, precision):
    return {"precision": precision}

def predict(model, image):
    w, h = image.size
    return {"boxes": np.array([[4, 4, w // 2, h // 2], [1, 1, 3, 3]]), "scores": np.array([0.9, 0.2]), "ids": [0, 1]}

def to_prediction(model, image, out):
    if FAIL_ON and image.size == FAIL_ON:
        return Boxes([[5, 5, 1, 1]], [0.9], ["cat"])
    return Boxes(out["boxes"], out["scores"], [["cat", "dog"][i] for i in out["ids"]])

def weight_files(size):
    return []
'''


@pytest.fixture
def repo(tmp_path):
    """A fake checkout with one model and a directory of local images."""
    model = tmp_path / "models" / "fake-det"
    model.mkdir(parents=True)
    (model / "adapter.py").write_text(FAKE_ADAPTER)
    images = tmp_path / "local"
    images.mkdir()
    Image.new("RGB", (64, 48), GREY).save(images / "street-01.png")
    Image.new("RGB", (80, 60), GREY).save(images / "park-02.jpg")
    (images / "notes.txt").write_text("not an image")
    return tmp_path


def test_cli_writes_one_webp_per_image(repo):
    model = repo / "models" / "fake-det"
    code = outputs.main([str(model), "--size", "s", "--task", "object-detection", "--images", str(repo / "local")])
    assert code == 0
    out = model / "outputs" / "s"
    assert sorted(p.name for p in out.iterdir()) == ["park-02.webp", "street-01.webp"]
    with Image.open(out / "park-02.webp") as im:
        assert im.size == (80, 60)
    log = (model / "outputs" / "s.log").read_text()
    assert "1 of 2 at score ≥ 0.5: cat" in log and "threshold: 0.5" in log


def test_cli_reads_the_task_from_the_adapter(repo):
    model = repo / "models" / "fake-det"
    (model / "adapter.py").write_text('TASK = "object-detection"\n' + FAKE_ADAPTER)
    assert outputs.main([str(model), "--size", "s", "--images", str(repo / "local")]) == 0
    (model / "adapter.py").write_text(FAKE_ADAPTER)
    assert outputs.main([str(model), "--size", "s", "--images", str(repo / "local")]) == 1


def test_cli_needs_images_for_an_image_model(repo):
    model = repo / "models" / "fake-det"
    assert outputs.main([str(model), "--size", "s", "--task", "object-detection"]) == 1


def test_cli_keeps_dev_outputs_out_of_the_bundle(repo):
    model = repo / "models" / "fake-det"
    dev = repo / "testset" / "_dev"
    dev.mkdir(parents=True)
    Image.new("RGB", (64, 48), GREY).save(dev / "coco-cats.jpg")
    assert outputs.main([str(model), "--size", "s", "--task", "object-detection", "--images", str(dev)]) == 0
    assert (dev / "outputs" / "fake-det" / "s" / "coco-cats.webp").is_file()
    assert not (model / "outputs").exists()
    code = outputs.main([str(model), "--size", "s", "--task", "object-detection", "--images", str(dev), "--out", str(model / "outputs" / "s")])
    assert code == 1 and not (model / "outputs").exists()


def test_cli_writes_nothing_if_any_image_fails(repo):
    model = repo / "models" / "fake-det"
    adapter = model / "adapter.py"
    adapter.write_text(FAKE_ADAPTER.replace("FAIL_ON = None", "FAIL_ON = (64, 48)"))
    code = outputs.main([str(model), "--size", "s", "--task", "object-detection", "--images", str(repo / "local")])
    assert code == 1
    assert not (model / "outputs" / "s").exists()


def test_cli_refuses_types_it_cannot_draw(repo):
    model = repo / "models" / "fake-det"
    assert outputs.main([str(model), "--size", "s", "--task", "ocr", "--images", str(repo / "local")]) == 1


def test_list_images(tmp_path):
    Image.new("RGB", (2, 2)).save(tmp_path / "Bad_Name.png")
    with pytest.raises(outputs.OutputsError, match="image-id"):
        outputs.list_images(tmp_path)
    (tmp_path / "Bad_Name.png").unlink()
    Image.new("RGB", (2, 2)).save(tmp_path / "a-1.png")
    Image.new("RGB", (2, 2)).save(tmp_path / "a-1.jpg")
    with pytest.raises(outputs.OutputsError, match="Two files"):
        outputs.list_images(tmp_path)
    assert outputs.is_dev(Path("/x/testset/_dev/a.jpg")) and not outputs.is_dev(Path("/x/testset/images/a.jpg"))


def test_coco_detections_map_names_and_keep_the_top_100():
    from mtr_measure import eval as ev

    cats = [{"id": 4, "name": "motorcycle"}, {"id": 17, "name": "cat"}]
    ids = ev.label_ids(cats, {"kitty": "cat"})
    assert ids["motorbike"] == 4 and ids["kitty"] == 17
    n = 150
    boxes = Boxes(np.tile([[10, 20, 30, 60]], (n, 1)), np.linspace(0.01, 0.99, n), ["cat"] * (n - 2) + ["N/A", "motorbike"])
    unknown: set[str] = set()
    out = ev.coco_detections(boxes, 7, ids, "bbox", unknown)
    assert len(out) == ev.MAX_DETS and not unknown
    assert out[0]["category_id"] == 4 and out[0]["score"] == 0.99  # the highest score first
    assert out[0]["bbox"] == [10.0, 20.0, 20.0, 40.0]  # xywh
    assert all(d["image_id"] == 7 for d in out)
    ev.coco_detections(Boxes([[0, 0, 1, 1]], [0.5], ["unicorn"]), 7, ids, "bbox", unknown)
    assert unknown == {"unicorn"}
