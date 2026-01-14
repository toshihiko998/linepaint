from __future__ import annotations
from dataclasses import dataclass
import torch
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

@dataclass
class PaintPass:
    pipe: StableDiffusionControlNetImg2ImgPipeline
    device: str

    @staticmethod
    def build(model_id: str, controlnet_id: str, device: str = "cuda") -> "PaintPass":
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        if device.startswith("cuda"):
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()

        pipe = pipe.to(device)
        return PaintPass(pipe=pipe, device=device)

    def run(
        self,
        line_img: Image.Image,
        prompt: str,
        negative: str,
        steps: int,
        cfg: float,
        strength: float,
        controlnet_scale: float,
        width: int,
        height: int,
        seed: int,
    ) -> Image.Image:
        # line image should drive structure (ControlNet). We also use it as init image.
        line_img = line_img.convert("RGB").resize((width, height), Image.LANCZOS)

        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=line_img,
            control_image=line_img,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            strength=float(strength),
            controlnet_conditioning_scale=float(controlnet_scale),
            generator=gen,
        ).images[0]

        return out

