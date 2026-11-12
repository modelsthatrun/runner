"""SDXL-Lightning by ByteDance, text-to-image, through diffusers. Size "4step" is the 4-step UNet checkpoint.

Follows the model card (https://huggingface.co/ByteDance/SDXL-Lightning): the distilled UNet loaded into
Stable Diffusion XL base 1.0's pipeline, float16 with the fp16 variant of the base weights, an Euler scheduler
with "trailing" timesteps, 4 steps and guidance 0. The card sets no size and seeds nothing, so the image is
the pipeline's default 1024x1024 and the generator is on the CPU. The UNet's config is read with load_config()
at the base repository's pinned commit; the card passes the repository id to from_config(), which does the same.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from mtr_measure import device

KIND = "generation"
TASK = "text-to-image"
SIZES = ["4step"]
PRECISIONS = ["fp16"]
PACKAGE = "diffusers"
GENERATION = {"4step": {"resolution": "1024x1024", "steps": 4, "guidance": 0.0, "generator": "cpu"}}
REPO = "ByteDance/SDXL-Lightning"
REVISION = "c9a24f48e1c025556787b0c58dd67a091ece2e44"
UNETS = {"4step": "sdxl_lightning_4step_unet.safetensors"}
# The text encoders, the VAE and the scheduler come from Stable Diffusion XL base 1.0.
BASE = "stabilityai/stable-diffusion-xl-base-1.0"
BASE_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
BASE_FILES = [
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def load(size: str, precision: str):
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from safetensors.torch import load_file

    dtype = DTYPES[precision]
    config = UNet2DConditionModel.load_config(BASE, subfolder="unet", revision=BASE_REVISION)
    unet = UNet2DConditionModel.from_config(config).to(device(), dtype)
    unet.load_state_dict(load_file(hf_hub_download(REPO, UNETS[size], revision=REVISION), device=device()))
    pipe = StableDiffusionXLPipeline.from_pretrained(BASE, revision=BASE_REVISION, unet=unet, torch_dtype=dtype, variant="fp16").to(device())
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return pipe


@torch.inference_mode()
def predict(model, x):
    return model(prompt=x.prompt, height=x.height, width=x.width, num_inference_steps=x.steps, guidance_scale=x.guidance,
                 generator=torch.Generator(x.generator).manual_seed(x.seed)).images[0]


def weight_files(size: str) -> list[Path]:
    unet = Path(hf_hub_download(REPO, UNETS[size], revision=REVISION))
    return [unet, *(Path(hf_hub_download(BASE, f, revision=BASE_REVISION)) for f in BASE_FILES)]
