"""Sana-Sprint by NVIDIA, text-to-image, through diffusers' SanaSprintPipeline. Size "1.6b" is its 1.6B transformer.

Follows the model card (https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers):
bfloat16 and 2 steps. The card sets nothing else, so the rest is the pipeline's defaults: 1024x1024 and
guidance 4.5, which this distilled model takes as an embedded value and not as a second pass. It seeds nothing,
so the generator is on the CPU. The pipeline's default complex_human_instruction, a fixed instruction it puts
before every prompt, stays as it is.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

KIND = "generation"
TASK = "text-to-image"
SIZES = ["1.6b"]
PRECISIONS = ["bf16"]
PACKAGE = "diffusers"
GENERATION = {"1.6b": {"resolution": "1024x1024", "steps": 2, "guidance": 4.5, "generator": "cpu"}}
REPO = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
REVISION = "19683c58b7ea290e55cedd8950ae1d86ada7ef96"
FILES = [
    "transformer/diffusion_pytorch_model.safetensors",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load(size: str, precision: str):
    from diffusers import SanaSprintPipeline

    pipe = SanaSprintPipeline.from_pretrained(REPO, revision=REVISION, torch_dtype=DTYPES[precision])
    return pipe.to("cuda")


@torch.inference_mode()
def predict(model, x):
    return model(prompt=x.prompt, height=x.height, width=x.width, num_inference_steps=x.steps, guidance_scale=x.guidance,
                 generator=torch.Generator(x.generator).manual_seed(x.seed)).images[0]


def weight_files(size: str) -> list[Path]:
    return [Path(hf_hub_download(REPO, f, revision=REVISION)) for f in FILES]
