import torch
from torch import nn
from diffusers import AutoencoderKL, UNet2DConditionModel

from .checkpoints import load_checkpoint, read_checkpoint
from .models.text import encode_prompt, load_text_encoders
from .models.vae import configure_vae
from .schedules import make_sched


class Contrastive_SDXL(nn.Module):
    def __init__(self, pretrained_path, device="cuda", torch_dtype=torch.float32,
                 pretrained_model_name_or_path="stabilityai/sdxl-turbo", caption="Driving at night."):
        super().__init__()

        sd = read_checkpoint(pretrained_path)

        tokenizers, text_encoders = load_text_encoders(pretrained_model_name_or_path, device)
        self.tokenizer_1 = tokenizers[0]
        self.tokenizer_2 = tokenizers[1]
        self.text_encoder_1 = text_encoders[0]
        self.text_encoder_2 = text_encoders[1]

        self.scheduler = make_sched(pretrained_model_name_or_path, device)
        self.timestep = self.scheduler.timesteps[0]

        self.vae = configure_vae(AutoencoderKL.from_pretrained(pretrained_model_name_or_path, 
                                                               subfolder="vae"),
                                                               initialize_skips=False)
        
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

        self.vae_enc, self.vae_dec = load_checkpoint(self.unet, self.vae, sd)

        self.caption = caption

        self.to(device=device)
        self.vae.to(dtype=torch.float32)
        self.unet.to(dtype=torch_dtype)
        self.text_encoder_1.to(dtype=torch_dtype)
        self.text_encoder_2.to(dtype=torch_dtype)


    @staticmethod
    def encode(img, vae_enc):
        return vae_enc(img)

    @staticmethod
    def decode(latents, vae_dec):
        return vae_dec(latents)

    @staticmethod
    def denoise(latents, unet, scheduler, timestep, prompt_embeds, pooled_embeds, time_ids):
        """
        Denoise latents using the UNet and scheduler
        """

        ts = torch.as_tensor(timestep, device=latents.device, dtype=torch.long).reshape(-1)
        if ts.numel() == 1:
            ts = ts.expand(latents.shape[0])

        eps_pred = unet(latents, ts, 
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids}).sample
        
        denoised_latent = torch.stack([scheduler.step(eps_pred[i], ts[i], 
                                                      latents[i], return_dict=True).prev_sample 
                                                      for i in range(latents.shape[0])])

        return denoised_latent

    @staticmethod
    def translate(x, vae_enc, unet, vae_dec,  sched, timesteps, prompt_embeds, pooled_embeds, time_ids):
        """
        Forward pass through the VAE encoder, UNet, and VAE decoder.
        Args:
            x (torch.Tensor): Input images (B, C, H, W)
            vae_enc (VAE_encode): VAE encoder wrapper
            unet (UNet2DConditionModel): LoRA-enhanced UNet model
            vae_dec (VAE_decode): VAE decoder wrapper
            sched: 1-step scheduler 
            timesteps (torch.LongTensor): Timestep tensor, shape (B,)
            text_emb (torch.Tensor): Text embedding (B, T, D) for conditioning
        Returns:
            torch.Tensor: Reconstructed / translated image (B, C, H, W)
        """

        encoder_dtype = next(vae_enc.parameters()).dtype
        decoder_dtype = next(vae_dec.parameters()).dtype
        unet_dtype = next(unet.parameters()).dtype

        latents = vae_enc(x.to(dtype=encoder_dtype))

        latents = latents.to(dtype=unet_dtype)
        prompt_embeds = prompt_embeds.to(device=latents.device, dtype=unet_dtype)
        pooled_embeds = pooled_embeds.to(device=latents.device, dtype=unet_dtype)
        time_ids = time_ids.to(device=latents.device, dtype=unet_dtype)

        denoised_latent = Contrastive_SDXL.denoise(
            latents, unet, sched, timesteps, prompt_embeds, pooled_embeds, time_ids)
        
        x_out = vae_dec(denoised_latent.to(dtype=decoder_dtype))

        return x_out

    def forward(self, x_t, caption=None, caption_emb=None, time_ids=None):
        """
        Contrastive-SDXL forward pass with dual-tokenizer and dual-encoder prompt processing 
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
    
        if caption_emb is None:
            prompt_embeds, pooled_embeds = encode_prompt(
                caption if caption is not None else self.caption, 
                self.tokenizer_1, 
                self.tokenizer_2, 
                self.text_encoder_1, 
                self.text_encoder_2)
        else:
            prompt_embeds, pooled_embeds = caption_emb

        B = x_t.shape[0]

        if prompt_embeds.shape[0] == 1:
            prompt_embeds = prompt_embeds.expand(B, -1, -1)
            pooled_embeds = pooled_embeds.expand(B, -1)

        if time_ids is None:
            H, W = x_t.shape[-2:]
            time_ids = torch.tensor([[H, W, 0, 0, H, W]], device=x_t.device)
        time_ids = time_ids.reshape(-1, 6).to(x_t.device).expand(B, -1)

        return self.translate(x_t, 
                               self.vae_enc, 
                               self.unet, 
                               self.vae_dec,
                               self.scheduler, 
                               self.timestep, 
                               prompt_embeds, 
                               pooled_embeds, 
                               time_ids)



def prepare_generator_paramters(unet, vae):
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
    
    unet.conv_in.requires_grad_(True)

    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])

    params_gen = list(unet.conv_in.parameters())

    for name, param in unet.named_parameters():
        if "lora" in name and "default" in name:
            params_gen.append(param)

    for name, param in vae.named_parameters():
        if "lora" in name and "vae_skip" in name:
            params_gen.append(param)

    for idx in range(1, 5):
        params_gen.extend(getattr(vae.decoder, f"skip_conv_{idx}").parameters())
        
    params_gen = list({id(param): param for param in params_gen}.values()) # remove duplicates

    return params_gen