from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from controlnet_aux import LineartDetector

@dataclass
class LinePass:
    pipe: StableDiffusionControlNetImg2ImgPipeline
    detector: LineartDetector
    device: str

    @staticmethod
    def build(model_id: str, controlnet_id: str, device: str = "cuda") -> "LinePass":
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
        detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
        return LinePass(pipe=pipe, detector=detector, device=device)

    def run(
        self,
        rough_img: Image.Image,
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
        rough_img = rough_img.convert("RGB").resize((width, height), Image.LANCZOS)

        # Control image = lineart extracted from rough
        control_img = self.detector(rough_img)
        control_img = control_img.convert("RGB")

        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=rough_img,
            control_image=control_img,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            strength=float(strength),
            controlnet_conditioning_scale=float(controlnet_scale),
            generator=gen,
        ).images[0]

        return out

