import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import torch_directml
dml = torch_directml.device()

model_id = "D:\stable-diffusion-v1-5"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


# disables safety checks to reduce memory usage
def disabled_safety_checker(images, clip_input):
    if len(images.shape)==4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False
pipe.safety_checker = disabled_safety_checker

pipe = pipe.to(dml)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, height=448, width=448).images[0]

image.save("astronaut_rides_horse.png")
