import os
import sys
import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import torch.nn.functional as F
p = "models/vae/"
sys.path.append(p)
from utils.ddpm_utils import make_1step_sched, make_4step_sched
from models.vae.custom_vae_forward import custom_vae_encoder_fwd, custom_vae_decoder_fwd
from diffusers import DDPMScheduler


class VAE_encode(nn.Module):
    """
    A wrapper around the VAE encoder used in SDXL-Turbo.

    This wrapper encodes an input image into the latent space,
    and caches the downsampled feature maps (down blocks) from the encoder
    for use in skip connections during decoding.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        # Forward pass through VAE encoder. The encoder caches down block features in `current_down_blocks`
        _ = self.vae.encode(x)  # trigger caching
        latent = self.vae.encode(x).latent_dist.sample()
        return latent * self.vae.config.scaling_factor  # Stable Diffusion expects latents in a specific numerical range, vae.config.scaling_factor adjusts these values so that the UNet can process them properly.


class VAE_decode(nn.Module):
    """
    A wrapper around the VAE decoder used in SDXL-Turbo.
    This decoder expects that encoder downsampled features have been cached.
    These cached features are injected into the decoder for skip connections.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        # pull out the cached down-blocks from the encoder
        skip_acts = self.vae.encoder.current_down_blocks
        assert skip_acts is not None, "Missing encoder features for skip connections"

        # clear the cache on the encoder immediately
        self.vae.encoder.current_down_blocks = None

        # inject into the decoder
        self.vae.decoder.incoming_skip_acts = skip_acts

        # do the decode (and ensure we never leave skip_acts around)
        try:
            x_rec = self.vae.decode(z / self.vae.config.scaling_factor).sample
        finally:
            # just in case the decoder also cached anything unexpected
            self.vae.decoder.incoming_skip_acts = None

        # clamp output back to valid image range
        return x_rec.clamp(-1, 1)


class ContrastiveSDXL_Turbo(nn.Module):
    """
    Contrastive image-to-image translation model using SDXL-Turbo as generator.

    Includes dual text encoders, LoRA-enabled UNet and VAE, and optional pretrained loading.
    """

    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder=None,
                 lora_rank_unet=32, lora_rank_vae=8):
        super().__init__()

        # Load both tokenizers for SDXL-Turbo
        self.tokenizer_1 = AutoTokenizer.from_pretrained("stabilityai/sdxl-turbo", subfolder="tokenizer")
        self.tokenizer_2 = AutoTokenizer.from_pretrained("stabilityai/sdxl-turbo", subfolder="tokenizer_2")

        # Load both text encoders
        self.text_encoder_1 = CLIPTextModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="text_encoder").cuda()
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("stabilityai/sdxl-turbo", subfolder="text_encoder_2").cuda()

        # One-step DDPM scheduler 
        self.scheduler = make_1step_sched()
        self.timestep = 239

        # Initialize VAE with LoRA adapters and skip connection layers
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-turbo", subfolder="vae")
        vae.encoder.forward = custom_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = custom_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False

        # Initialize UNet with LoRA adapters
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="unet")
        self.unet, self.vae = unet, vae

        # Load from pretrained checkpoint if provided
        state_dict = torch.load(pretrained_path)
        self.load_ckpt_from_state_dict(state_dict)

        self.caption = "Driving at night."

        # Move core components to GPU
        self.vae_enc.cuda()
        self.vae_dec.cuda()
        self.unet.cuda()

    def load_ckpt_from_state_dict(self, sd):
        """
        Loads a model checkpoint and reconstructs all LoRA adapters and their weights
        for both the UNet and VAE components, along with encoder/decoder state dicts.

        Parameters:
            sd (dict): A dictionary-like state dict containing:
                - LoRA adapter configurations (target modules, ranks)
                - LoRA adapter weights for encoder, decoder, other blocks
                - State dicts for wrapped VAE_encode and VAE_decode modules
        """

        # === UNET ADAPTERS SETUP ===
        # Build LoRA configs for encoder, decoder, and other blocks using metadata from checkpoint
        lora_conf_encoder = LoraConfig(
            r=sd["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=sd["l_target_modules_encoder"],
            lora_alpha=sd["rank_unet"]
        )

        lora_conf_decoder = LoraConfig(
            r=sd["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=sd["l_target_modules_decoder"],
            lora_alpha=sd["rank_unet"]
        )

        lora_conf_others = LoraConfig(
            r=sd["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=sd["l_modules_others"],
            lora_alpha=sd["rank_unet"]
        )

        # Add adapters individually only if they don't exist
        if not hasattr(self.unet, "peft_config") or "default_encoder" not in self.unet.peft_config:
            self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        if not hasattr(self.unet, "peft_config") or "default_decoder" not in self.unet.peft_config:
            self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        if not hasattr(self.unet, "peft_config") or "default_others" not in self.unet.peft_config:
            self.unet.add_adapter(lora_conf_others, adapter_name="default_others")

        # Load LoRA weights for encoder adapter
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_encoder.weight", ".weight")
            if "lora" in n and "default_encoder" in n:
                p.data.copy_(sd["sd_encoder"][name_sd])

        # Load LoRA weights for decoder adapter
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_decoder.weight", ".weight")
            if "lora" in n and "default_decoder" in n:
                p.data.copy_(sd["sd_decoder"][name_sd])

        # Load LoRA weights for other blocks (e.g., mid block, ff layers)
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_others.weight", ".weight")
            if "lora" in n and "default_others" in n:
                p.data.copy_(sd["sd_other"][name_sd])

        # Set active adapters
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # === VAE ADAPTER SETUP ===
        # Build VAE LoRA config
        vae_lora_config = LoraConfig(
            r=sd["rank_vae"],
            init_lora_weights="gaussian",
            target_modules=sd["vae_lora_target_modules"]
        )

        if not hasattr(self.vae, "peft_config") or "vae_skip" not in self.vae.peft_config:
            self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")


        # Apply decoder gamma if needed for skip scaling
        self.vae.decoder.gamma = 1

        # === RESTORE ENCODER & DECODER STATE ===
        # Rewrap encoder/decoder and load their fine-tuned weights
        self.vae_enc = VAE_encode(self.vae)
        self.vae_enc.load_state_dict(sd["sd_vae_enc"])

        self.vae_dec = VAE_decode(self.vae)
        self.vae_dec.load_state_dict(sd["sd_vae_dec"])

    @staticmethod
    def encode(img, vae_enc):
        """
        Encode pixels → latents, caching skip-connections for the decoder.
        Returns:
            latents: a tensor of shape (B, C, H', W')
        """
        # inside VAE_encode.forward you already sample & scale:
        return vae_enc(img)
    
    @staticmethod
    def decode(latents, vae_dec):
        """
        Decode latents → pixels, clearing skip caches immediately after injection.
        Returns:
            images: (B, 3, H, W) in [-1,1]
        """
        return vae_dec(latents)

    @staticmethod
    def denoise(latents: torch.FloatTensor,
            unet: nn.Module,
            scheduler,
            timestep, 
            prompt_embeds: torch.FloatTensor,
            pooled_embeds: torch.FloatTensor,
            time_ids: torch.Tensor): 
        """
        Denoise latents using the UNet and scheduler
        """

        B = latents.shape[0]

        ts = torch.full((B,), timestep, device=latents.device, dtype=torch.long)

        noisy_lantent = latents

        model_out = unet(
                noisy_lantent, ts,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids},
            )
            
        eps_pred = model_out.sample

        #  Denoised‑latent  
        denoised_latent  = scheduler.step(eps_pred, ts, noisy_lantent, return_dict=True).pred_original_sample         
        return denoised_latent, None
    
    @staticmethod
    def get_trainable_params(unet, vae):
        """
        Collects and returns all trainable parameters in the UNet and VAE.

        Specifically includes:
        - LoRA adapter weights in UNet (for encoder, decoder, and others)
        - LoRA adapter weights in VAE ("vae_skip")
        - Learnable skip connection conv layers in VAE decoder

        Args:
            unet (UNet2DConditionModel): The SDXL-Turbo UNet with LoRA adapters
            vae (AutoencoderKL): The VAE module with LoRA adapters and skip convs

        Returns:
            List[torch.nn.Parameter]: List of all trainable parameters
        """
        # params_gen = []  # for a clean LoRA-only setup
        params_gen = list(unet.conv_in.parameters())
        unet.conv_in.requires_grad_(True)

        # --- UNET ---
        # Ensure UNet is using all expected adapters
        unet.set_adapters(["default_encoder", "default_decoder", "default_others"])

        for name, param in unet.named_parameters():
            if "lora" in name and "default" in name:
                assert param.requires_grad
                params_gen.append(param)

        # --- VAE ---
        for name, param in vae.named_parameters():
            if "lora" in name and "vae_skip" in name:
                assert param.requires_grad
                params_gen.append(param)

        # Add skip conv layers from decoder
        params_gen += list(vae.decoder.skip_conv_1.parameters())
        params_gen += list(vae.decoder.skip_conv_2.parameters())
        params_gen += list(vae.decoder.skip_conv_3.parameters())
        params_gen += list(vae.decoder.skip_conv_4.parameters())

        def dedupe_params(params):
            seen = set()
            unique = []
            for p in params:
                if id(p) not in seen:
                    seen.add(id(p))
                    unique.append(p)
            return unique

        params_gen = dedupe_params(params_gen)
        return params_gen


    # ==================================================Inference Code=============================================================

    @staticmethod
    def forward_with_networks(x, vae_enc, unet, vae_dec, sched, timesteps, prompt_embeds, pooled_embeds, time_ids, denoise_steps):
        """
        Forward pass through the VAE encoder, UNet, and VAE decoder.
    
        Args:
            x (torch.Tensor): Input images (B, C, H, W)
            vae_enc (VAE_encode): VAE encoder wrapper
            unet (UNet2DConditionModel): LoRA-enhanced UNet model
            vae_dec (VAE_decode): VAE decoder wrapper
            sched: 1-step scheduler (e.g., DDIM scheduler)
            timesteps (torch.LongTensor): Timestep tensor, shape (B,)
            text_emb (torch.Tensor): Text embedding (B, T, D) for conditioning
    
        Returns:
            torch.Tensor: Reconstructed / translated image (B, C, H, W)
        """


        latents = vae_enc(x).to(x.dtype)

        B = latents.shape[0]

        ts = torch.full((B,), timesteps, device=latents.device, dtype=torch.long)

        noisy_lantent = latents

        eps_pred = unet(
                noisy_lantent, ts,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids},
            ).sample

        #  Denoised‑latent pass 
        denoised_latent  = sched.step(eps_pred, ts, noisy_lantent, return_dict=True).pred_original_sample  
        x_out = vae_dec(denoised_latent)                

        return x_out         
        
    def forward(self, x_t, caption=None, caption_emb=None, time_ids=None, denoise_steps=4):
        """
        SDXL-Turbo forward pass with dual-tokenizer and dual-encoder prompt processing 
        NOTE: only used in inference.py file.
    
        This method performs the following:
        - Encodes a caption using both SDXL-Turbo tokenizers and text encoders.
        - Extracts token-level embeddings (from penultimate hidden states) and concatenates them.
        - Extracts a pooled/global embedding (from text_encoder_2 output) for global context.
        - Feeds the resulting embeddings to the image-to-image translation pipeline.
    
        Args:
            x_t (torch.Tensor): Input image tensor of shape (B, C, H, W).
            caption (str, optional): Text prompt to condition the generation. Ignored if `caption_emb` is provided.
            caption_emb (Tuple[torch.Tensor, torch.Tensor], optional):
                Precomputed (prompt_embeds, pooled_prompt_embeds).
                Use this to skip text encoding and reuse previous embeddings.
    
        Returns:
            torch.Tensor: Output image after translation, same shape as input (B, C, H, W).
        """
    
        if caption_emb is not None:
            # Use provided precomputed embeddings: (token-level, pooled-level)
            prompt_embeds, pooled_embeds = caption_emb
        else:
            # If no caption string is passed, fall back to default caption set at init
            if caption is None:
                assert self.caption is not None, "No caption provided and no default set."
                caption = self.caption
    
            # Tokenize the caption using both SDXL tokenizers
            tokens_1 = self.tokenizer_1(
                caption,
                max_length=self.tokenizer_1.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(x_t.device)
    
            tokens_2 = self.tokenizer_2(
                caption,
                max_length=self.tokenizer_2.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(x_t.device)

            # Get encoder outputs with all hidden states
            out_1 = self.text_encoder_1(tokens_1, output_hidden_states=True)
            out_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)

            # Extract the penultimate hidden state from each encoder and concatenate
            # This is the official SDXL behavior for conditioning on token-level features
            prompt_embeds = torch.cat([
                out_1.hidden_states[-2],  # text_encoder_1 hidden state
                out_2.hidden_states[-2],  # text_encoder_2 hidden state
            ], dim=-1).detach().clone()
    
            # Extract pooled embedding from text_encoder_2 (standard SDXL global summary)
            pooled_embeds = out_2[0]  # shape: (B, D)

        # Run full pipeline: VAE encode → UNet → Scheduler step → VAE decode
        return self.forward_with_networks(
            x_t,
            self.vae_enc,
            self.unet,
            self.vae_dec,
            self.scheduler,
            self.timestep,
            prompt_embeds,
            pooled_embeds,
            time_ids, 
            denoise_steps=denoise_steps
        )

