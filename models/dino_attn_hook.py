import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


class DINOAttentionHook:
    def __init__(self, model, nce_layers=None):
        """
        Hooks into specific transformer blocks of DINOv2 to capture token embeddings for hDCE/SRC losses.

        Args:
            model: The DINOv2 Vision Transformer model
            nce_layers: List of block indices to hook into 
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.handles = []
        self.attn_features = []
        self.nce_layers = nce_layers or list(range(100))
        self._register_hooks()

        # Freeze DINOv2 parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def _register_hooks(self):
        self.global_layer_idx = 0
        for idx, block in enumerate(self.model.blocks):
            if idx in self.nce_layers:
                def make_hook(i):
                    def _hook_fn(module, input, output):
                        # output: [B, N, C]
                        if output.ndim == 3:
                            B, N, C = output.shape
                            spatial = int(N ** 0.5)
                            if (spatial * spatial)+1 != N:
                                print(f"Skipping non-square attention map at layer {i}, shape={output.shape}")
                                return
                            feat = output[:, 1:, :] if N > 1 and output.shape[1] == N else output  # Remove CLS if exists
                            feat = feat.permute(0, 2, 1).reshape(B, C, spatial, spatial)  # [B, C, H, W]
                            self.attn_features.append(feat.detach().to(self.device))
                    return _hook_fn
                handle = block.register_forward_hook(make_hook(i=idx))
                self.handles.append(handle)

    def clear(self):
        self.attn_features = []

    def get_and_clear_features(self):
        feats = self.attn_features
        self.clear()
        return feats

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.clear()

    def extract_attention_features(self, img: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through DINOv2 model to collect token embeddings from hooked layers.

        Args:
            img (torch.Tensor): Input image tensor of shape [B, 3, H, W]

        Returns:
            list[torch.Tensor]: Collected features as [B, C, H, W]
        """
        self.clear()
        _ = self.model(img)
        return self.get_and_clear_features()
    



def resize_for_dino(img: torch.Tensor, size: int = 504) -> torch.Tensor:
    """
    Resize a single image tensor [1, 3, H, W] to [1, 3, size, size] for DINOv2.

    Args:
        img (torch.Tensor): Image tensor of shape [3, H, W], values in [-1, 1]
        size (int): Target height and width (default is 504)

    Returns:
        torch.Tensor: Resized image of shape [1, 3, size, size]
    """

    # Resize to [1, 3, size, size]
    resized = F.resize(img, size=size, interpolation=InterpolationMode.BICUBIC)

    return resized