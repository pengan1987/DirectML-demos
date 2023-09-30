from diffusers import DiffusionPipeline
from datetime import datetime

import torch
import torch_directml

dml = torch_directml.device()
pipeline = DiffusionPipeline.from_pretrained("d:\\tiny_sd",torch_dtype=torch.float16)
pipeline = pipeline.to(dml)
prompt = input("Enter your prompt: ")
image = pipeline(prompt).images[0]

current_datetime = datetime.now()

# Format the date and time in YYYYMMDDHHMMSS format
formatted_datetime = current_datetime.strftime('%Y%m%d%H%M%S')
image.save(formatted_datetime+".png")