"""Adapter for <model name>, a text-to-image model: how the shared tool loads and runs it (docs/adapters.md).

`mtr check <id> --size <size>` tries it, `mtr run <id> --size <size>` times one image from the task's timing
prompt, and `mtr outputs <id> --size <size>` makes an image for every prompt of the text-to-image prompt set
(docs/protocol.md, "Image generation"). Follow the official model card: its pipeline, dtype, resolution, steps
and guidance. Every component stays on the GPU: leave out enable_model_cpu_offload() and similar calls, which
model cards add only to save memory. A model that doesn't fit without them doesn't run on this GPU.
"""

from pathlib import Path

import torch

KIND = "generation"
TASK = "text-to-image"
SIZES = ["base"]  # checkpoint sizes, named as the model's own card names them
PRECISIONS = ["bf16"]  # what the official code runs; list bf16 and leave fp16 out if the card loads it in bfloat16
PACKAGE = "diffusers"  # the installed distribution that runs the pipeline; its version is recorded
# Per size, from the model card's example: resolution (WxH), denoising steps, guidance scale, and the device
# its torch.Generator is seeded on. The tool passes them to predict() as x.steps, x.guidance, x.width, x.height.
GENERATION = {"base": {"resolution": "1024x1024", "steps": 28, "guidance": 3.5, "generator": "cpu"}}
REPO = "owner/model"
REVISION = "0000000000000000000000000000000000000000"  # the repository's 40-hex commit, never a branch


def load(size: str, precision: str):
    """The pipeline at this precision, every component on the GPU."""
    raise NotImplementedError
    # from diffusers import DiffusionPipeline
    # dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    # return DiffusionPipeline.from_pretrained(REPO, revision=REVISION, torch_dtype=dtype).to("cuda")


@torch.inference_mode()
def predict(model, x):
    """One image, as the model card makes it. `x` is a mtr_measure.generation.Generate."""
    raise NotImplementedError
    # return model(x.prompt, width=x.width, height=x.height, num_inference_steps=x.steps, guidance_scale=x.guidance,
    #              generator=torch.Generator(x.generator).manual_seed(x.seed)).images[0]


def weight_files(size: str) -> list[Path]:
    """Every weight file load() read, the denoiser's first. Records carry their sha256."""
    raise NotImplementedError
