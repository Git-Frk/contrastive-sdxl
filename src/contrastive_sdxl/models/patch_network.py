import torch
from torch import nn


def init_weights(network, init_type="normal", init_gain=0.02):
    """Initialize network weights."""
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if init_type == "normal":
                nn.init.normal_(module.weight, 0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(module.weight, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(module.weight, gain=init_gain)
            else:
                raise ValueError(f"Unknown initialization: {init_type}")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1, init_gain)
            nn.init.zeros_(module.bias)

    network.apply(initialize)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1 / self.power)
        return x.div(norm + 1e-7)


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type="normal", init_gain=0.02, nc=256):
        super().__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc
        self.init_type = init_type
        self.init_gain = init_gain
        self.mlp_init = False

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            mlp = nn.Sequential(nn.Linear(feat.shape[1], self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc))
            setattr(self, f"mlp_{mlp_id}", mlp)
        init_weights(self, self.init_type, self.init_gain)
        self.mlp_init = True

    def forward(self, feats, num_patches, patch_ids=None):
        if not feats:
            raise ValueError("No DINO features were captured.")
        
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)

        return_feats, return_ids = [], []
        for feat_id, feat in enumerate(feats):
            B, _, H, W = feat.shape
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is None:
                    patch_id = torch.randperm(H * W, device=feat.device)[:min(num_patches, H * W)]
                else:
                    patch_id = patch_ids[feat_id]
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                patch_id = []
                x_sample = feat_reshape
            if self.use_mlp:
                mlp = getattr(self, f"mlp_{feat_id}").to(device=x_sample.device)
                if not torch.is_autocast_enabled(x_sample.device.type):
                    mlp = mlp.to(dtype=x_sample.dtype)
                x_sample = mlp(x_sample)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape(B, x_sample.shape[-1], H, W)

            return_feats.append(x_sample)
            return_ids.append(patch_id)
        return return_feats, return_ids


def initialise_patch_projector(patch_projector, init_type="xavier", init_gain=0.02, args=None):
    if patch_projector not in ("sample", "mlp_sample"):
        raise ValueError(f"Unknown projection model: {patch_projector}")
    network = PatchSampleF(patch_projector == "mlp_sample", init_type, init_gain, args.patch_projector_nc)
    init_weights(network, init_type, init_gain)
    return network
