# from torch import nn
# from peft import LoraConfig
# from diffusers import AutoencoderKL


# vae_target_modules = [
#     "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
#     "to_k", "to_q", "to_v", "to_out.0",
# ]


# def custom_vae_encoder_fwd(self, sample):
#     sample = self.conv_in(sample)
#     down_blocks_feature = []

#     for down_block in self.down_blocks:
#         down_blocks_feature.append(sample)
#         sample = down_block(sample)
 
#     sample = self.mid_block(sample)
#     sample = self.conv_norm_out(sample)
#     sample = self.conv_act(sample)
#     sample = self.conv_out(sample)
#     self.current_down_blocks = down_blocks_feature
#     return sample


# def custom_vae_decoder_fwd(self, sample, latent_embeds=None):
#     sample = self.conv_in(sample)
#     sample = self.mid_block(sample, latent_embeds)
#     sample = sample.to(next(self.up_blocks.parameters()).dtype)

#     for idx, up_block in enumerate(self.up_blocks):
#         if not self.ignore_skip:
#             skip_conv = getattr(self, f"skip_conv_{idx + 1}")
#             skip_in = skip_conv(self.incoming_skip_activations[::-1][idx] * self.gamma)
#             sample = sample + skip_in
#         sample = up_block(sample, latent_embeds)
    
#     sample = self.conv_norm_out(sample) if latent_embeds is None else self.conv_norm_out(sample, latent_embeds)
    
#     sample = self.conv_act(sample)
#     sample = self.conv_out(sample)
#     return sample


# def configure_vae(vae, initialize_skips=True):
#     vae.encoder.forward = custom_vae_encoder_fwd.__get__(vae.encoder)
#     vae.decoder.forward = custom_vae_decoder_fwd.__get__(vae.decoder)
#     device, dtype = next(vae.parameters()).device, next(vae.parameters()).dtype
#     for idx, (cin, cout) in enumerate(((512, 512), (256, 512), (128, 512), (128, 256)), 1):
#         conv = nn.Conv2d(cin, cout, 1, bias=False).to(device=device, dtype=dtype)
#         if initialize_skips:
#             nn.init.constant_(conv.weight, 1e-5)
#         setattr(vae.decoder, f"skip_conv_{idx}", conv)
#     vae.decoder.ignore_skip = False
#     vae.decoder.gamma = 1
#     vae.encoder.current_down_blocks = None
#     vae.decoder.incoming_skip_activations = None
#     return vae


# def add_vae_adapter(vae, rank, target_modules):
#     if "vae_skip" not in getattr(vae, "peft_config", {}):
#         config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=target_modules)
#         print("🔧 Adding LoRA adapter to VAE")
#         vae.add_adapter(config, adapter_name="vae_skip")    
#     else:
#         print("✅ Adapter 'vae_skip' already present, skipping")
#     vae.set_adapter("vae_skip")


# def setup_vae(torch_dtype, rank=8, return_lora_module_names=False,
#                    pretrained_model_name_or_path="stabilityai/sdxl-turbo"):

#     vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch_dtype)
#     configure_vae(vae)
#     vae.requires_grad_(True)
#     vae.train()
#     target_modules = vae_target_modules
#     add_vae_adapter(vae, rank, target_modules)
#     for name in ("skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4"):
#         getattr(vae.decoder, name).requires_grad_(True) # Unfreeze skip convolutions after LoRA setup freezes non-LoRA weights.
#     return (vae, target_modules) if return_lora_module_names else vae


# class VAE_encode(nn.Module):
#     def __init__(self, vae):
#         super().__init__()
#         self.vae = vae

#     def forward(self, x):
#         # Preserve both encoder calls from the original generator.
#         self.vae.encode(x)
#         latent = self.vae.encode(x).latent_dist.sample()
#         return latent * self.vae.config.scaling_factor # Stable Diffusion expects latents in a specific numerical range, vae.config.scaling_factor adjusts these values so that the UNet can process them properly.


# class VAE_decode(nn.Module):
#     def __init__(self, vae):
#         super().__init__()
#         self.vae = vae

#     def forward(self, z):
#         skip_acts = self.vae.encoder.current_down_blocks
#         if skip_acts is None:
#             raise RuntimeError("Encode an image before decoding; VAE skip features are missing.")
#         self.vae.encoder.current_down_blocks = None
#         self.vae.decoder.incoming_skip_activations = skip_acts
#         try:
#             x_reconstruct = self.vae.decode(z / self.vae.config.scaling_factor).sample
#         finally:
#             self.vae.decoder.incoming_skip_activations = None
#         return x_reconstruct.clamp(-1, 1)



# ===============================test code for VAE skip connections with LoRA adapter===============================

from torch import nn
from peft import LoraConfig
from diffusers import AutoencoderKL


vae_target_modules = [
    "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
    "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
    "to_k", "to_q", "to_v", "to_out.0",
]


def custom_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    down_blocks_feature = []

    for down_block in self.down_blocks:
        down_blocks_feature.append(sample)
        sample = down_block(sample)
 
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = down_blocks_feature
    return sample


def custom_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(next(self.up_blocks.parameters()).dtype)

    for idx, up_block in enumerate(self.up_blocks):
        if not self.ignore_skip:
            skip_conv = getattr(self, f"skip_conv_{idx + 1}")
            skip_in = skip_conv(self.incoming_skip_activations[::-1][idx] * self.gamma)
            sample = sample + skip_in
        sample = up_block(sample, latent_embeds)
    
    sample = self.conv_norm_out(sample) if latent_embeds is None else self.conv_norm_out(sample, latent_embeds)
    
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def configure_vae(vae, initialize_skips=True):
    vae.encoder.forward = custom_vae_encoder_fwd.__get__(vae.encoder)
    vae.decoder.forward = custom_vae_decoder_fwd.__get__(vae.decoder)
    device, dtype = next(vae.parameters()).device, next(vae.parameters()).dtype
    for idx, (cin, cout) in enumerate(((512, 512), (256, 512), (128, 512), (128, 256)), 1):
        conv = nn.Conv2d(cin, cout, 1, bias=False).to(device=device, dtype=dtype)
        if initialize_skips:
            nn.init.constant_(conv.weight, 1e-5)
        setattr(vae.decoder, f"skip_conv_{idx}", conv)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    vae.encoder.current_down_blocks = None
    vae.decoder.incoming_skip_activations = None
    return vae


def add_vae_adapter(vae, rank, target_modules):
    if "vae_skip" not in getattr(vae, "peft_config", {}):
        config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=target_modules)
        print("🔧 Adding LoRA adapter to VAE")
        vae.add_adapter(config, adapter_name="vae_skip")    
    else:
        print("✅ Adapter 'vae_skip' already present, skipping")
    vae.set_adapter("vae_skip")


def setup_vae(torch_dtype, rank=8, return_lora_module_names=False,
                   pretrained_model_name_or_path="stabilityai/sdxl-turbo"):

    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch_dtype)
    configure_vae(vae)
    vae.requires_grad_(False)
    vae.train()
    
    target_modules = vae_target_modules
    add_vae_adapter(vae, rank, target_modules)

    return (vae, target_modules) if return_lora_module_names else vae


class VAE_encode(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        # Preserve both encoder calls from the original generator.
        self.vae.encode(x)
        latent = self.vae.encode(x).latent_dist.sample()
        return latent * self.vae.config.scaling_factor # Stable Diffusion expects latents in a specific numerical range, vae.config.scaling_factor adjusts these values so that the UNet can process them properly.


class VAE_decode(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        skip_acts = self.vae.encoder.current_down_blocks
        if skip_acts is None:
            raise RuntimeError("Encode an image before decoding; VAE skip features are missing.")
        self.vae.encoder.current_down_blocks = None
        self.vae.decoder.incoming_skip_activations = skip_acts
        try:
            x_reconstruct = self.vae.decode(z / self.vae.config.scaling_factor).sample
        finally:
            self.vae.decoder.incoming_skip_activations = None
        return x_reconstruct.clamp(-1, 1)
