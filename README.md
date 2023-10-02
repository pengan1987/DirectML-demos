# PyTorch-DirectML inference example
This repository are examples showing how to using PyTorch-DirectML to run some generative models. To give potential developers a quick hands on example of platform neutral AI applications.

## General notice
The `requirements.txt` contains dependencies to run all the scripts, not all the scripts need all the dependencies. So, If you want to make your own deployment, you can try to remove some of them.

## Model specific notices
### microsoft/phi-1_5
Microsoft Phi-1.5 is a LLM model can generate Q&A text etc. The tricky part to run this model is you can't just use `.to(dml)` to move the model to DirectML GPU, but need `torch.set_default_device(dml)` to use the DirectML as default inference device.

Also you might stuck with `transformers==4.30.2`, I have tried with `transformers==4.33.3` but got an error, check the transformers [issue #26512](https://github.com/huggingface/transformers/issues/26512) for details.

**Tested Hardware**
- [OK] Ryzen 7 5800H/16GB RAM/Vega 8 iGPU(GCN5)
- [OK] i5 7200U/20GB RAM/UHD620 iGPU(Gen9.5)

### Segmind/tiny-sd
This is a distilled text-to-image model can generate Stable Diffusion like images with smaller hardware resources. This model is easy to migrate, just set `pipeline.to(dml)` then it can run on DirectML GPUs.

**Tested Hardware**
- [OK] Ryzen 7 5800H/16GB RAM/Vega 8 iGPU(GCN5)
- [OK] i5 7200U/20GB RAM/UHD620 iGPU(Gen9.5)

### runwayml/stable-diffusion-v1-5
This is the popular Stable Diffusion model, easy to migrate, just set `pipeline.to(dml)`

**Tested Hardware**
- [OK] Ryzen 7 5800H/16GB RAM/Vega 8 iGPU(GCN5)
- [Limited] i5 7200U/20GB RAM/UHD620 iGPU(Gen9.5), 9.9GB shared VRAM available, out of VRAM when generating 512x512 image, 448x448 works.
