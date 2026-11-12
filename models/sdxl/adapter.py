"""Stable Diffusion XL by Stability AI, text-to-image, through diffusers. Size "base" is SDXL base 1.0.

Follows the model card (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0): float16 with the
fp16 variant of the weights, pipe(prompt=...) with the pipeline's defaults, which are 1024x1024, 50 steps and
guidance 5.0. The card seeds nothing, so the generator is on the CPU, as diffusers recommends for
reproducible images. The base model alone, without the optional refiner.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from mtr_measure import device

KIND = "generation"
TASK = "text-to-image"
SIZES = ["base"]
PRECISIONS = ["fp16"]
PACKAGE = "diffusers"
GENERATION = {"base": {"resolution": "1024x1024", "steps": 50, "guidance": 5.0, "generator": "cpu"}}
REPOS = {"base": ("stabilityai/stable-diffusion-xl-base-1.0", "462165984030d82259a11f4367a4eed129e94a7b")}
FILES = [
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def load(size: str, precision: str):
    from diffusers import DiffusionPipeline

    repo, revision = REPOS[size]
    pipe = DiffusionPipeline.from_pretrained(repo, revision=revision, torch_dtype=DTYPES[precision], use_safetensors=True, variant="fp16")
    return pipe.to(device())


@torch.inference_mode()
def predict(model, x):
    return model(prompt=x.prompt, height=x.height, width=x.width, num_inference_steps=x.steps, guidance_scale=x.guidance,
                 generator=torch.Generator(x.generator).manual_seed(x.seed)).images[0]


def weight_files(size: str) -> list[Path]:
    repo, revision = REPOS[size]
    return [Path(hf_hub_download(repo, f, revision=revision)) for f in FILES]
