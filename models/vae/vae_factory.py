import os
import sys
import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from custom_vae_forward import custom_vae_encoder_fwd, custom_vae_decoder_fwd


def initialize_vae(rank=8, return_lora_module_names=False, torch_dtype=torch.float32):
    """
    Initializes the VAE from SDXL-Turbo with LoRA adapters and custom skip connections.
    Also overrides the encoder and decoder forward methods with custom ones.
    """
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-turbo", subfolder="vae", torch_dtype=torch_dtype)
    vae.requires_grad_(False)

    # Patch forward methods
    vae.encoder.forward = custom_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = custom_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

    vae.requires_grad_(True)
    vae.train()

    # Add skip connection convolutional layers
    vae.decoder.skip_conv_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_2 = nn.Conv2d(256, 512, kernel_size=1, bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_3 = nn.Conv2d(128, 512, kernel_size=1, bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_4 = nn.Conv2d(128, 256, kernel_size=1, bias=False).cuda().requires_grad_(True)

    # Initialize skip conv weights to small values
    nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)

    # Enable skip usage during decoding
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1

    # Define modules for LoRA injection
    l_vae_target_modules = [
        "conv1", "conv2", "conv_in", "conv_shortcut",
        "conv", "conv_out", "skip_conv_1", "skip_conv_2",
        "skip_conv_3", "skip_conv_4", "to_k", "to_q",
        "to_v", "to_out.0"
    ]

    vae_lora_config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules)
    
    # Prevent duplicate adapter creation
    if not hasattr(vae, "peft_config") or "vae_skip" not in vae.peft_config:
        print("🔧 Adding LoRA adapter to VAE")
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    
    else:
        print("✅ Adapter 'vae_skip' already present, skipping")

    # Activate it
    vae.set_adapter("vae_skip")
    

    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae