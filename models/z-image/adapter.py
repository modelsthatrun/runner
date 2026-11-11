"""Z-Image by Tongyi-MAI, text-to-image, through diffusers' ZImagePipeline.

Size "turbo" is Z-Image-Turbo, distilled to 8 DiT passes; "base" is Z-Image, the undistilled model.
Follows the model cards (https://huggingface.co/Tongyi-MAI/Z-Image-Turbo and Tongyi-MAI/Z-Image): bfloat16,
a generator seeded on CUDA, SDPA attention and no compilation (both optional on the cards). Turbo makes
1024x1024 in 9 steps (8 DiT passes) at guidance 0. Base makes the card's 720x1280 in 50 steps at guidance 4,
with an empty negative prompt and cfg_normalization off, as its card passes them. Turbo's weights are stored
in float32 and loaded as bfloat16, as the card does.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

KIND = "generation"
TASK = "text-to-image"
SIZES = ["turbo", "base"]
PRECISIONS = ["bf16"]
PACKAGE = "diffusers"
GENERATION = {
    "turbo": {"resolution": "1024x1024", "steps": 9, "guidance": 0.0, "generator": "cuda"},
    "base": {"resolution": "720x1280", "steps": 50, "guidance": 4.0, "generator": "cuda"},
}
REPOS = {
    "turbo": ("Tongyi-MAI/Z-Image-Turbo", "f332072aa78be7aecdf3ee76d5c247082da564a6"),
    "base": ("Tongyi-MAI/Z-Image", "04cc4abb7c5069926f75c9bfde9ef43d49423021"),
}
TRANSFORMER_SHARDS = {"turbo": 3, "base": 2}
# What base's card passes besides the settings in GENERATION. Turbo's card passes nothing more.
EXTRA = {"turbo": {}, "base": {"negative_prompt": "", "cfg_normalization": False}}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load(size: str, precision: str):
    from diffusers import ZImagePipeline

    repo, revision = REPOS[size]
    pipe = ZImagePipeline.from_pretrained(repo, revision=revision, torch_dtype=DTYPES[precision], low_cpu_mem_usage=False)
    pipe.extra = EXTRA[size]
    return pipe.to("cuda")


@torch.inference_mode()
def predict(model, x):
    return model(prompt=x.prompt, height=x.height, width=x.width, num_inference_steps=x.steps, guidance_scale=x.guidance,
                 generator=torch.Generator(x.generator).manual_seed(x.seed), **model.extra).images[0]


def weight_files(size: str) -> list[Path]:
    repo, revision = REPOS[size]
    shards = TRANSFORMER_SHARDS[size]
    files = [
        *(f"transformer/diffusion_pytorch_model-0000{i}-of-0000{shards}.safetensors" for i in range(1, shards + 1)),
        *(f"text_encoder/model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)),
        "vae/diffusion_pytorch_model.safetensors",
    ]
    return [Path(hf_hub_download(repo, f, revision=revision)) for f in files]
