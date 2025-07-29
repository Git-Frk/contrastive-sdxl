import os
import sys
import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict


def initialize_unet(torch_dtype, rank, return_lora_module_names=False):
    """
    Initializes the SDXL-Turbo UNet with LoRA adapters for encoder, decoder, and other modules.
    Collects and categorizes parameter names for fine-tuning using LoRA.
    """
    unet = UNet2DConditionModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="unet", torch_dtype=torch_dtype)
    unet.requires_grad_(False)
    unet.train()

    # Separate target modules by block type
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = [
        "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
        "conv_in", "conv_shortcut", "conv_out",
        "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"
    ]


    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight", ""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight", ""))
                break

    # Create LoRA configs
    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_encoder,
                                   lora_alpha=rank)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_decoder,
                                   lora_alpha=rank)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_modules_others,
                                  lora_alpha=rank)
    
    # Add adapters individually only if they don't exist, with console feedback
    cfg = getattr(unet, "peft_config", {})

    if "default_encoder" not in cfg:
        print("🔧 Adding LoRA adapter 'default_encoder' to UNet")
        unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    else:
        print("✅ Adapter 'default_encoder' already present, skipping")

    if "default_decoder" not in cfg:
        print("🔧 Adding LoRA adapter 'default_decoder' to UNet")
        unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    else:
        print("✅ Adapter 'default_decoder' already present, skipping")

    if "default_others" not in cfg:
        print("🔧 Adding LoRA adapter 'default_others' to UNet")
        unet.add_adapter(lora_conf_others, adapter_name="default_others")
    else:
        print("✅ Adapter 'default_others' already present, skipping")

    # Activate all adapters
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])

    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet
