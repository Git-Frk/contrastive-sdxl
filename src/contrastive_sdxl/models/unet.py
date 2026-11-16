from peft import LoraConfig
from diffusers import UNet2DConditionModel


adapter_names = ["default_encoder", "default_decoder", "default_others"]


def add_unet_adapters(unet, rank, target_modules):

    for adapter_name, modules in zip(adapter_names, target_modules):
        if adapter_name not in getattr(unet, "peft_config", {}):
            config = LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=modules)
            print(f"🔧 Adding LoRA adapter '{adapter_name}' to UNet")
            unet.add_adapter(config, adapter_name=adapter_name)
        else:
            print(f"✅ Adapter '{adapter_name}' already present, skipping")

    unet.set_adapters(list(adapter_names))


def setup_unet(torch_dtype, rank, return_lora_module_names=False,
                    pretrained_model_name_or_path="stabilityai/sdxl-turbo"):

    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", torch_dtype=torch_dtype)
    unet.requires_grad_(False)
    unet.train()

    lora_encoder, lora_decoder, lora_others = [], [], []
    patterns = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in",
                "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    
    for name, param in unet.named_parameters():
        if "bias" in name or "norm" in name or not any(pattern in name for pattern in patterns):
            continue
        if "down_blocks" in name or "conv_in" in name:
            modules = lora_encoder
        elif "up_blocks" in name:
            modules = lora_decoder
        else:
            modules = lora_others
        modules.append(name.replace(".weight", ""))

    target_modules = [lora_encoder, lora_decoder, lora_others]
    add_unet_adapters(unet, rank, target_modules)

    if return_lora_module_names:
        return unet, lora_encoder, lora_decoder, lora_others
    else:
        return unet

