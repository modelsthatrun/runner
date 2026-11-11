"""PixArt-Sigma by the PixArt team, text-to-image, through diffusers' PixArtSigmaPipeline. Size "xl-2-1024-ms" is PixArt-Sigma-XL/2 at 1024 px.

Follows the model card (https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS): float16, safetensors
weights and pipe(prompt) with the pipeline's defaults, which are 1024x1024, 20 steps and guidance 4.5, with
caption cleaning on. The card seeds nothing, so the generator is on the CPU. The T5 text encoder's weights are
stored in float32 and loaded as float16, as the card does.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

KIND = "generation"
TASK = "text-to-image"
SIZES = ["xl-2-1024-ms"]
PRECISIONS = ["fp16"]
PACKAGE = "diffusers"
GENERATION = {"xl-2-1024-ms": {"resolution": "1024x1024", "steps": 20, "guidance": 4.5, "generator": "cpu"}}
REPO = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
REVISION = "e102b3591cc82e97071b8b4cb90d834d0c487207"
FILES = [
    "transformer/diffusion_pytorch_model.safetensors",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def load(size: str, precision: str):
    from diffusers import PixArtSigmaPipeline

    pipe = PixArtSigmaPipeline.from_pretrained(REPO, revision=REVISION, torch_dtype=DTYPES[precision], use_safetensors=True)
    return pipe.to("cuda")


@torch.inference_mode()
def predict(model, x):
    return model(prompt=x.prompt, height=x.height, width=x.width, num_inference_steps=x.steps, guidance_scale=x.guidance,
                 generator=torch.Generator(x.generator).manual_seed(x.seed)).images[0]


def weight_files(size: str) -> list[Path]:
    return [Path(hf_hub_download(REPO, f, revision=REVISION)) for f in FILES]
