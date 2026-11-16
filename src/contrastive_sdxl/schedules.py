import torch
from torch import nn
from diffusers import DDPMScheduler


def make_sched(pretrained_model_name_or_path="stabilityai/sdxl-turbo", device="cuda"):
    scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    scheduler.set_timesteps(1, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    return scheduler


class LinearSchedule(nn.Module):
    """
    Args:
        start: lambda value at start_step
        end: lambda value at max_steps
        max_steps: step where ramp finishes
        start_step: step where ramp begins (lambda = 0 before this)
    """
    def __init__(self, start=0.01, end=1.0, max_steps=12000, start_step=0):
        super().__init__()
        if max_steps <= start_step:
            raise ValueError("max_steps must be greater than start_step")
        self.start = start
        self.end = end
        self.max_steps = max_steps
        self.start_step = start_step

    def forward(self, step):
        step_t = torch.as_tensor(step, dtype=torch.float32)
        frac = ((step_t - self.start_step) / (self.max_steps - self.start_step)).clamp(0, 1)
        value = self.start + frac * (self.end - self.start)
        return torch.where(step_t < self.start_step, torch.zeros_like(value), value)
