import os
import requests
from tqdm import tqdm
from diffusers import DDPMScheduler


def make_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sdxl-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda") # this is usually used for inference and not training
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step

