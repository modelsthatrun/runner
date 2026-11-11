"""Playground v2.5 by Playground, text-to-image, through diffusers. Size "1024px-aesthetic" is its one released checkpoint.

Follows the model card (https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic): float16 with
the fp16 variant of the weights, the scheduler the repository ships (the card's DPM++ 2M Karras swap is
optional and left out), 50 steps and guidance 3. The card sets no size and seeds nothing, so the image is the
pipeline's default 1024x1024 and the generator is on the CPU.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

KIND = "generation"
TASK = "text-to-image"
SIZES = ["1024px-aesthetic"]
PRECISIONS = ["fp16"]
PACKAGE = "diffusers"
GENERATION = {"1024px-aesthetic": {"resolution": "1024x1024", "steps": 50, "guidance": 3.0, "generator": "cpu"}}
REPO = "playgroundai/playground-v2.5-1024px-aesthetic"
REVISION = "1e032f13f2fe6db2dc49947dbdbd196e753de573"
FILES = [
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def load(size: str, precision: str):
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(REPO, revision=REVISION, torch_dtype=DTYPES[precision], variant="fp16")
    return pipe.to("cuda")


@torch.inference_mode()
def predict(model, x):
    return model(prompt=x.prompt, height=x.height, width=x.width, num_inference_steps=x.steps, guidance_scale=x.guidance,
                 generator=torch.Generator(x.generator).manual_seed(x.seed)).images[0]


def weight_files(size: str) -> list[Path]:
    return [Path(hf_hub_download(REPO, f, revision=REVISION)) for f in FILES]
