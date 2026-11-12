"""Adapter for LT-DETR (LTDETRv2 by Lightly AG), real-time object detection on COCO's 80 classes.

LT-DETR has no repository or paper of its own: it ships inside the lightly-train package, built on
EdgeCrafter's ECViT backbones. Its docs (docs/source/object_detection.md) load a COCO checkpoint with
lightly_train.load_model("ltdetrv2-s-coco") and call model.predict(image). predict() resizes the
image to 640x640 (a plain resize, no letterbox), runs the model in deploy mode, keeps the top 300
queries and returns boxes in the input image's pixels, with labels as class ids into model.classes.

- Threshold. predict() defaults to 0.6, above the task's display threshold of 0.5, so predict()
  passes 0.5. eval_predict() passes 0.0, the default of lightly_train.benchmark_object_detection.
- Precision. predict() runs in the parameters' dtype, fp32 as loaded. The torch backend of
  benchmark_object_detection offers "fp16-mixed": torch.autocast(float16) around the forward pass
  (lightly_train/_commands/benchmark_backends.py). fp16 runs the same way here. Autocast wraps the
  documented predict() call, and its pre- and post-processing hold no ops that autocast casts.
- Usage events. predict() queues an "inference_started" event with system information and sends it
  from a background thread. The README's opt-out, LIGHTLY_TRAIN_EVENTS_DISABLED=1, is set below,
  so no network call happens during timed calls.
- Sizes. Only the three COCO-pretrained sizes are listed; ltdetrv2-x has no COCO checkpoint. The
  DINOv2 and DINOv3 backbone LTDETR models are a different family and are not listed here.
- ONNX Runtime on the CPU. The docs' export notebook (examples/notebooks/object_detection_export.ipynb at
  v0.18.0): model.export_onnx(out="model.onnx") with its defaults, then a default ort.InferenceSession, the
  pre-processing it builds from the file's metadata (a resize of the PIL image to the graph's 640x640, ToTensor,
  the stored normalization), and its postprocess_detections(): sigmoid, the top num_queries query-class pairs, the
  threshold, cxcywh to xyxy in the image's pixels. predict() passes the display threshold, as for the PyTorch row.
"""

import os
from pathlib import Path

import torch

from mtr_measure import device, weights_dir
from mtr_measure.predictions import Boxes

# lightly_train downloads named checkpoints into LIGHTLY_TRAIN_MODEL_CACHE_DIR (default
# ~/.cache/lightly-train/models), keeping the file name from its URL.
CACHE = weights_dir("lt-detr")
os.environ.setdefault("LIGHTLY_TRAIN_MODEL_CACHE_DIR", str(CACHE))
os.environ.setdefault("LIGHTLY_TRAIN_EVENTS_DISABLED", "1")

import lightly_train  # noqa: E402

TASK = "object-detection"
SIZES = ["s", "m", "l"]
PRECISIONS = ["fp16", "fp32"]
PACKAGE = "lightly-train"
INPUT = {"s": "640x640", "m": "640x640", "l": "640x640"}
NAMES = {"s": "ltdetrv2-s-coco", "m": "ltdetrv2-m-coco", "l": "ltdetrv2-l-coco"}
# The checkpoint files behind each name (lightly_train/_task_models/ltdetr_object_detection/config.py).
FILES = {
    "s": "edgecrafter_ecvitt_ltdetr_coco_260624_f8aefe49.pt",
    "m": "edgecrafter_ecvittplus_ltdetr_coco_260720_723357e9.pt",
    "l": "edgecrafter_ecvits_ltdetr_coco_260720_62ff87cd.pt",
}
DISPLAY_THRESHOLD = 0.5
ONNXRUNTIME = {"status": "runs-as-is"}


def load(size, precision):
    model = lightly_train.load_model(NAMES[size], device=device())
    model.deploy()  # predict() would do this on its first call
    return {"model": model, "fp16": precision == "fp16"}


def _predict(m, x, threshold):
    with torch.autocast(device(), dtype=torch.float16, enabled=m["fp16"]):
        return m["model"].predict(x, threshold=threshold)


def predict(m, x):
    if "sess" in m:
        return ort_predict(m, x, DISPLAY_THRESHOLD)
    return _predict(m, x, DISPLAY_THRESHOLD)


def eval_predict(m, x):
    return _predict(m, x, 0.0)


def weight_files(size):
    return [CACHE / FILES[size]]


def onnx_file(size, out):
    """The notebook's export of the released checkpoint, with export_onnx()'s defaults."""
    model = lightly_train.load_model(NAMES[size], device=device())
    path = Path(out) / "model.onnx"
    model.export_onnx(out=str(path))
    return {"onnx": path, "exporter": f"lightly-train=={lightly_train.__version__} model.export_onnx", "args": ["out=model.onnx"]}


def load_onnx(size, onnx_path):
    """The notebook's ONNX Runtime setup: the classes, normalization and input size from the file, and a default session."""
    import json

    import onnx
    import onnxruntime as ort
    import torchvision.transforms.v2 as T

    onnx_model = onnx.load(str(onnx_path))
    metadata = {entry.key: entry.value for entry in onnx_model.metadata_props}
    classes = {int(k): v for k, v in json.loads(metadata["classes"]).items()}
    image_normalize = json.loads(metadata["image_normalize"]) if "image_normalize" in metadata else None
    images_input = onnx_model.graph.input[0].type.tensor_type.shape.dim
    image_size = (images_input[2].dim_value, images_input[3].dim_value)  # (H, W)
    transforms = T.Compose([T.Resize(image_size), T.ToTensor(), T.Normalize(**image_normalize) if image_normalize else T.Identity()])
    sess = ort.InferenceSession(str(onnx_path))
    dtype = {"tensor(float)": "float32", "tensor(float16)": "float16"}[sess.get_inputs()[0].type]
    return {"sess": sess, "transforms": transforms, "classes": classes, "dtype": dtype}


def onnx_session(m):
    return m["sess"]


def ort_predict(m, x, threshold):
    from torchvision.ops import box_convert

    w, h = x.size
    logits, boxes = m["sess"].run(output_names=None, input_feed={"images": m["transforms"](x)[None].numpy().astype(m["dtype"])})
    logits, boxes = torch.from_numpy(logits).float(), torch.from_numpy(boxes).float()
    # The notebook's postprocess_detections().
    scores = logits[0].sigmoid()
    num_classes = scores.shape[-1]
    boxes_xyxy = box_convert(boxes[0], in_fmt="cxcywh", out_fmt="xyxy")
    scores, flat_index = scores.flatten().topk(logits.shape[1])
    keep = scores > threshold
    scores, flat_index = scores[keep], flat_index[keep]
    query_index, class_index = flat_index // num_classes, flat_index % num_classes
    boxes = boxes_xyxy[query_index] * torch.tensor([w, h, w, h], dtype=boxes_xyxy.dtype)
    class_ids = list(m["classes"].keys())
    labels = torch.tensor([class_ids[int(i)] for i in class_index])
    return {"labels": labels, "bboxes": boxes, "scores": scores}


def to_prediction(m, x, out):
    names = m["classes"] if "classes" in m else m["model"].classes
    # As in D-FINE, whose decoder these checkpoints use, a low-scoring box can come back with a negative
    # width or height (one of the top 300 on COCO image 104782 for m, scoring 0.008, at fp16 and
    # fp32 alike). Collapse it to zero area: it still counts as a false positive in COCO's
    # evaluation, and never as a match.
    boxes = out["bboxes"].float().cpu()
    boxes[:, 2:] = boxes[:, 2:].maximum(boxes[:, :2])
    return Boxes(
        boxes.numpy(),
        out["scores"].float().cpu().numpy(),
        [names[int(i)] for i in out["labels"]],
    )
