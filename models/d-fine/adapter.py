"""Adapter for D-FINE, real-time object detection on COCO's 80 classes.

D-FINE has no package. The adapter runs a pinned checkout of the official repository, the way its
tools/inference/torch_inf.py does: the config's model and postprocessor in deploy mode, the image
resized to 640x640 with torchvision, and the top 300 boxes back in the input image's pixels. That
script and the repository's COCO evaluation run in fp32; FP16 exists only through TensorRT, so
PRECISIONS is fp32. Only the COCO-trained checkpoints are listed: the README says the Objects365
ones may be bound by the Objects365 dataset terms.

Its TensorRT row follows the README's "Deployment" steps: tools/deployment/export_onnx.py on the
released checkpoint, then `trtexec --onnx=model.onnx --saveEngine=model.engine --fp16`, which the tool
reproduces with TensorRT's Python API (mtr_measure/trt.py). tools/inference/trt_inf.py's TRTInference
runs the engine, with the same 640x640 resize as torch_inf.py. The export puts the postprocessor in the
graph, so the engine returns what forward() and postprocess() return together.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

from mtr_measure import weights_dir
from mtr_measure.predictions import Boxes

REPO_URL = "https://github.com/Peterande/D-FINE.git"
COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
WEIGHTS = weights_dir("d-fine")
REPO = WEIGHTS / "D-FINE"
RELEASE = "https://github.com/Peterande/storage/releases/download/dfinev1.0"

TASK = "object-detection"
SIZES = ["n", "s", "m", "l", "x"]
PRECISIONS = ["fp32"]
INPUT = {s: "640x640" for s in SIZES}
# The README's display threshold in tools/inference/torch_inf.py.
THRESHOLD = 0.4
# The README's TensorRT step: trtexec with --fp16 and nothing else.
TENSORRT = {"precision": "fp16", "builder": {"fp16": True}, "status": "runs-as-is"}


def package_version():
    return f"Peterande/D-FINE@{COMMIT}"


def checkout():
    if not (REPO / ".git").is_dir():
        subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO)], check=True)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "origin"], check=True)
        subprocess.run(["git", "-C", str(REPO), "checkout", "-q", COMMIT], check=True)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def load(size, precision):
    checkout()
    from src.core import YAMLConfig

    ckpt = checkpoint_file(size)
    cfg = YAMLConfig(str(REPO / "configs" / "dfine" / f"dfine_hgnetv2_{size}_coco.yml"), resume=str(ckpt))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(ckpt, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    return {"net": Model().to("cuda"), "transforms": T.Compose([T.Resize((640, 640)), T.ToTensor()]), "names": coco_names()}


def coco_names():
    from src.data.dataset.coco_dataset import mscoco_category2name, mscoco_label2category

    return [mscoco_category2name[mscoco_label2category[i]] for i in range(80)]


def checkpoint_file(size):
    ckpt = weight_files(size)[0]
    if not ckpt.is_file():
        torch.hub.download_url_to_file(f"{RELEASE}/{ckpt.name}", str(ckpt))
    return ckpt


def export_onnx(size, out):
    """The README's export, run as it is on the released checkpoint. It writes the ONNX file next to the
    checkpoint it is given, so it gets a link to the checkpoint in `out`."""
    checkout()
    ckpt = checkpoint_file(size)
    link = Path(out) / ckpt.name
    link.symlink_to(ckpt)
    args = ["--check", "-c", f"configs/dfine/dfine_hgnetv2_{size}_coco.yml", "-r", ckpt.name]
    subprocess.run([sys.executable, "tools/deployment/export_onnx.py", *args[:-1], str(link)], cwd=REPO, check=True)
    return {"onnx": link.with_suffix(".onnx"), "exporter": f"Peterande/D-FINE@{COMMIT} tools/deployment/export_onnx.py", "args": args}


def load_engine(size, engine):
    """The engine in the official TRTInference, as tools/inference/trt_inf.py's main() makes it."""
    checkout()
    spec = importlib.util.spec_from_file_location("dfine_trt_inf", REPO / "tools" / "inference" / "trt_inf.py")
    trt_inf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trt_inf)
    return {"trt": trt_inf.TRTInference(str(engine), device="cuda:0"), "transforms": T.Compose([T.Resize((640, 640)), T.ToTensor()]),
            "names": coco_names()}


def trt_call(m, x):
    # tools/inference/trt_inf.py's process_image(): the blob, then the engine's three outputs for the one image.
    w, h = x.size
    blob = {"images": m["transforms"](x)[None].to("cuda"), "orig_target_sizes": torch.tensor([w, h])[None].to("cuda")}
    out = m["trt"](blob)
    return out["labels"][0].cpu(), out["boxes"][0].cpu(), out["scores"][0].cpu()


@torch.inference_mode()
def eval_predict(m, x):
    # The top 300 boxes with every score, as the repository's COCO evaluation scores them.
    if "trt" in m:
        return trt_call(m, x)
    w, h = x.size
    labels, boxes, scores = m["net"](m["transforms"](x).unsqueeze(0).to("cuda"), torch.tensor([[w, h]], device="cuda"))
    return labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()


# predict() in the three phases `mtr profile` times apart: the tensor on the GPU, the network, and the decoded result.
@torch.inference_mode()
def preprocess(m, x):
    return m["transforms"](x).unsqueeze(0).to("cuda")


@torch.inference_mode()
def forward(m, images):
    return m["net"].model(images)


@torch.inference_mode()
def postprocess(m, raw, x):
    w, h = x.size
    labels, boxes, scores = m["net"].postprocessor(raw, torch.tensor([[w, h]], device="cuda"))
    labels, boxes, scores = labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()
    keep = scores > THRESHOLD
    return labels[keep], boxes[keep], scores[keep]


@torch.inference_mode()
def predict(m, x):
    if "trt" in m:
        labels, boxes, scores = trt_call(m, x)
        keep = scores > THRESHOLD
        return labels[keep], boxes[keep], scores[keep]
    return postprocess(m, forward(m, preprocess(m, x)), x)


def weight_files(size):
    return [WEIGHTS / f"dfine_{size}_coco.pth"]


def to_prediction(m, x, out):
    labels, boxes, scores = out
    # A few of the top 300 come back with a negative width or height (152 of 90,000 on 300 COCO
    # images for x, all scoring under 0.02). Collapse them to zero area: they still count as false
    # positives in COCO's evaluation, and never as matches.
    boxes = boxes.clone()
    boxes[:, 2:] = boxes[:, 2:].maximum(boxes[:, :2])
    return Boxes(boxes.numpy(), scores.numpy(), [m["names"][int(i)] for i in labels])
