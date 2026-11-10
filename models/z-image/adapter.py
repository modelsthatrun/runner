"""Z-Image by Tongyi-MAI, text-to-image, through diffusers' ZImagePipeline. Size "turbo" is Z-Image-Turbo.

Follows the model card (https://huggingface.co/Tongyi-MAI/Z-Image-Turbo): bfloat16, 1024x1024, 9 steps (8 DiT
passes), guidance 0, a generator seeded on CUDA, SDPA attention and no compilation (both optional on the card).
The weights are stored in float32 and loaded as bfloat16, as the card does.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

KIND = "generation"
TASK = "text-to-image"
SIZES = ["turbo"]
PRECISIONS = ["bf16"]
PACKAGE = "diffusers"
GENERATION = {"turbo": {"resolution": "1024x1024", "steps": 9, "guidance": 0.0, "generator": "cuda"}}
REPOS = {"turbo": ("Tongyi-MAI/Z-Image-Turbo", "f332072aa78be7aecdf3ee76d5c247082da564a6")}
FILES = [
    *(f"transformer/diffusion_pytorch_model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)),
    *(f"text_encoder/model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)),
    "vae/diffusion_pytorch_model.safetensors",
]
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load(size: str, precision: str):
    from diffusers import ZImagePipeline

    repo, revision = REPOS[size]
    pipe = ZImagePipeline.from_pretrained(repo, revision=revision, torch_dtype=DTYPES[precision], low_cpu_mem_usage=False)
    return pipe.to("cuda")


@torch.inference_mode()
def predict(model, x):
    return model(prompt=x.prompt, height=x.height, width=x.width, num_inference_steps=x.steps, guidance_scale=x.guidance,
                 generator=torch.Generator(x.generator).manual_seed(x.seed)).images[0]


def weight_files(size: str) -> list[Path]:
    repo, revision = REPOS[size]
    return [Path(hf_hub_download(repo, f, revision=revision)) for f in FILES]
