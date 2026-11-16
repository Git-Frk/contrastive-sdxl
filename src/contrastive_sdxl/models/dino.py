from math import isqrt

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def load_dino(device, repo=None):
    if repo:
        model = torch.hub.load(str(repo), "dinov2_vitg14", source="local")
    else:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14", trust_repo=True)
    return model.to(device)


class DINOAttentionHook:
    """Capture block outputs while retaining gradients with respect to the image."""

    def __init__(self, model, nce_layers=None):
        self.model = model
        self.nce_layers = list(range(len(model.blocks))) if nce_layers is None else list(nce_layers)
        if not self.nce_layers or any(i < 0 or i >= len(model.blocks) for i in self.nce_layers):
            raise ValueError("nce_layers must identify existing DINO transformer blocks")
        self.handles = []
        self.attn_features = []
        self.model.requires_grad_(False)
        self.model.eval()
        for idx, block in enumerate(model.blocks):
            if idx in self.nce_layers:
                self.handles.append(block.register_forward_hook(self._capture))

    def _capture(self, module, inputs, output):
        if output.ndim != 3:
            raise ValueError(f"Expected [B, N, C], got {tuple(output.shape)}")
        B, N, C = output.shape
        spatial = isqrt(N - 1)
        if spatial * spatial != N - 1:
            raise ValueError(f"Expected square patch tokens plus CLS, got {tuple(output.shape)}")
        feat = output[:, 1:, :].permute(0, 2, 1).reshape(B, C, spatial, spatial)
        self.attn_features.append(feat)

    def clear(self):
        self.attn_features = []

    def get_and_clear_features(self):
        feats = self.attn_features
        self.clear()
        return feats

    def extract_attention_features(self, img):
        """ 
        Forward pass through DINOv2 model to collect token embeddings from hooked layers.
        Args:
            img (torch.Tensor): Input image tensor of shape [B, 3, H, W]
        Returns:
            list[torch.Tensor]: Collected features as [B, C, H, W]
        """
        self.clear()
        try:
            self.model(img)
            return self.get_and_clear_features()
        except Exception:
            self.clear()
            raise

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()


def resize_for_dino(img, size=504):
    # Resize a single image tensor [1, 3, H, W] to [1, 3, size, size] for DINOv2.
    # Retain the original [-1, 1] inputs and bicubic resize.
    return F.resize(img, size=size, interpolation=InterpolationMode.BICUBIC)
