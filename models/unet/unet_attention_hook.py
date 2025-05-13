import torch
from torch import nn


class UNetAttentionHook:
    def __init__(self, model):
        """
        Hooks into SDXL-Turbo UNet's down_blocks to capture all attention maps
        from BasicTransformerBlock.attn1 modules across CrossAttnDownBlock2D layers.

        Args:
            model: The UNet2DConditionModel instance to hook into.
        """
        self.model = model
        # Determine device from model parameters
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            # Fallback if model has no parameters
            self.device = torch.device('cpu')
        self.handles = []
        # Storage for collected feature maps per forward pass
        self.attn_features = []  # list of torch.Tensor [B, C, H, W]
        self._register_hooks()

#     def _register_hooks(self):
#         # Attach forward hooks on each attention sub-layer
#         for block in self.model.down_blocks:
#             if hasattr(block, "attentions"):
#                 for attn_mod in block.attentions:
#                     for transformer in attn_mod.transformer_blocks:
#                         handle = transformer.attn1.register_forward_hook(self._hook_fn)
#                         self.handles.append(handle)

#     def _hook_fn(self, module, input, output):
#         """
#         Captures the attention output:
#         - output: Tensor of shape [B, T, C] (tokens)
#         Reshapes to [B, C, H, W] for square T.
#         """
#         attn_map = output  # [B, T, C]
#         B, T, C = attn_map.shape
#         spatial = int(T ** 0.5)
#         if spatial * spatial != T:
#             # Skip non-square token maps
#             return
#         # reshape tokens to spatial map
#         attn_map = attn_map.permute(0, 2, 1).reshape(B, C, spatial, spatial)
#         # Detach and move to stored device
#         self.attn_features.append(attn_map.detach().to(self.device))

    def _register_hooks(self):
        self.global_layer_idx = 0  # reset global index
        for block in self.model.down_blocks:
            if hasattr(block, "attentions"):
                for attn_mod in block.attentions:
                    for transformer in attn_mod.transformer_blocks:
                        idx = self.global_layer_idx
                        if idx in self.nce_layers:  # Register only if needed

                            def make_hook(i):
                                def _hook_fn(module, input, output):
                                    B, T, C = output.shape
                                    spatial = int(T ** 0.5)
                                    if spatial * spatial != T:
                                        return
                                    attn_map = output.permute(0, 2, 1).reshape(B, C, spatial, spatial)
                                    self.attn_features.append(attn_map.detach().to(self.device))

                                return _hook_fn

                            handle = transformer.attn1.register_forward_hook(make_hook(i=idx))
                            self.handles.append(handle)
                        self.global_layer_idx += 1

    def clear(self):
        """
        Clears collected attention features before a new forward pass.
        """
        self.attn_features = []

    def get_and_clear_features(self):
        """
        Returns the collected attention maps and resets internal storage.

        Returns:
            list[torch.Tensor]: List of attention maps [B, C, H, W]
        """
        feats = self.attn_features
        self.clear()
        return feats

    def remove(self):
        """
        Removes all registered hooks and clears stored features.
        """
        for handle in self.handles:
            handle.remove()
        self.clear()

    def extract_attention_features(
        self,
        latent: torch.Tensor,
        unet: nn.Module,
        timestep: torch.LongTensor,
        text_embed: torch.Tensor,
        pooled_embed: torch.Tensor,
        time_ids: torch.Tensor,
        nce_layers: list[int]
    ) -> list[torch.Tensor]:
        """
        Encodes `x`, runs a UNet forward under hooks, and returns only the requested layers.

        Args:
            x: Input images [B, 3, H, W]
            vae_enc: VAE encoder wrapper
            unet: SDXL UNet model
            timestep: [B] timesteps
            text_embed: [B, T, D] token embeddings
            pooled_embed: [B, D] pooled text embedding
            nce_layers: indices of layers to select from the collected features

        Returns:
            list[torch.Tensor]: selected attention feature maps
        """
        # 1) clear previous
        self.clear()

        # 2) forward UNet (hooks capture features)
        _ = unet(
            latent, timestep=timestep,
            encoder_hidden_states=text_embed,
            added_cond_kwargs={"text_embeds": pooled_embed,
                               "time_ids": time_ids},
        )
        # 4) retrieve and select
        all_feats = self.get_and_clear_features()
        # guard index range
        selected = [all_feats[i] for i in nce_layers if 0 <= i < len(all_feats)]
        return selected
