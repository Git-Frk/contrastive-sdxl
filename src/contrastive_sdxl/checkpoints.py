from pathlib import Path
from collections.abc import Mapping
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

import torch

from .models.unet import adapter_names, add_unet_adapters
from .models.vae import VAE_encode, VAE_decode, add_vae_adapter


required_keys = {
    "rank_unet", "rank_vae", "l_target_modules_encoder", "l_target_modules_decoder",
    "l_modules_others", "sd_encoder", "sd_decoder", "sd_other", "vae_lora_target_modules",
    "sd_vae_enc", "sd_vae_dec", "sd_unet_conv_in"
}


def read_checkpoint(model_path):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Expected one checkpoint file: {model_path}")
    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(sd, Mapping):
        raise ValueError("Checkpoint must contain a state dictionary.")
    missing = required_keys - sd.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {', '.join(sorted(missing))}")
    return sd


def load_checkpoint(unet, vae, sd):
    
    #  UNET ADAPTERS SETUP 
    targets = (sd["l_target_modules_encoder"], sd["l_target_modules_decoder"], sd["l_modules_others"])
    add_unet_adapters(unet, sd["rank_unet"], targets)
    for adapter_name, key in zip(adapter_names, ("sd_encoder", "sd_decoder", "sd_other")):
        expected = get_peft_model_state_dict(unet, adapter_name=adapter_name)
        missing = expected.keys() - sd[key].keys()
        unexpected = sd[key].keys() - expected.keys()
        if missing or unexpected:
            raise ValueError(f"Invalid {key}: missing tensors {sorted(missing)}, unexpected tensors {sorted(unexpected)}")
        set_peft_model_state_dict(unet, sd[key], adapter_name=adapter_name)
    unet.conv_in.load_state_dict(sd["sd_unet_conv_in"])

    # VAE ADAPTER SETUP 
    add_vae_adapter(vae, sd["rank_vae"], sd["vae_lora_target_modules"])
    vae_enc, vae_dec = VAE_encode(vae), VAE_decode(vae)
    vae_enc.load_state_dict(sd["sd_vae_enc"], strict=True)
    vae_dec.load_state_dict(sd["sd_vae_dec"], strict=True)
    return vae_enc, vae_dec


def save_checkpoint(path, unet, vae_enc, vae_dec, args, unet_target_modules, vae_lora_target_modules):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder, decoder, others = unet_target_modules

    sd = {
        "rank_unet": args.lora_rank_unet, 
        "rank_vae": args.lora_rank_vae,
        "l_target_modules_encoder": encoder, 
        "l_target_modules_decoder": decoder,
        "l_modules_others": others, 
        "vae_lora_target_modules": vae_lora_target_modules,
        "sd_vae_enc": vae_enc.state_dict(), 
        "sd_vae_dec": vae_dec.state_dict()
        
        }
    for name, key in zip(adapter_names, ("sd_encoder", "sd_decoder", "sd_other")):
        sd[key] = get_peft_model_state_dict(unet, adapter_name=name)

    sd["sd_unet_conv_in"] = unet.conv_in.state_dict()

    torch.save(sd, path)
