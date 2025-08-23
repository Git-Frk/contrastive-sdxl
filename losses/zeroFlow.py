import torch
import torch.nn.functional as F
import math


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

def coarse_identity_grid(batch, Hc, Wc, device):
    """
    Create a coarse identity grid used for feature matching.
    
    The grid spans normalized coordinates in the range [-1, 1] over Wc and Hc.
    This is the reference identity warp: used for comparing with the imgB warp.

    Args:
        batch (int): Batch size.
        Hc (int): Height of the coarse feature map.
        Wc (int): Width of the coarse feature map.
        device (torch.device): Device to place the tensor on.

    Returns:
        torch.Tensor: Identity grid of shape (B, Hc, Wc, 2)
                      with normalized coordinates.
    """
    u = torch.linspace(-1.0, 1.0, Wc, device=device)
    v = torch.linspace(-1.0, 1.0, Hc, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")               # Wc×Hc
    id_grid = torch.stack([uu, vv], dim=-1)                    # Wc×Hc×2
    id_grid = id_grid.permute(1, 0, 2)                         # Hc×Wc×2
    id_grid = id_grid.expand(batch, *id_grid.shape)           # B×Hc×Wc×2
    return id_grid


def charbonnier(res2, eps2, alpha):
    """
    Generalized Charbonnier loss. Robust loss function used for residuals.

    Args:
        res2 (Tensor): Squared residuals (typically ‖Δ‖²), shape (…).
        eps2 (float): Epsilon squared for numerical stability.
        alpha (float): Exponent for controlling the robustness.

    Returns:
        Tensor: Robustified residual, same shape as res2.
    """
    return (res2 + eps2).pow(alpha)

# ------------------------------------------------------------------
# Loss Module
# ------------------------------------------------------------------
class IdentityWarpLoss(torch.nn.Module):
    """
    Penalises any geometric drift introduced by the generator.

    • Uses both halves of RoMa's symmetric warp (A→B and B→A).
    • Per-pixel weight is RoMa's certainty **after** sigmoid.
    • Optional certainty floor ignores very ambiguous regions.
    """

    def __init__(
        self,
        alpha: float = 0.45,
        eps: float = 1e-3,
        cert_thresh: float | None = None,   # to drop sky / water
    ):
        super().__init__()
        self.alpha = alpha
        self.eps2 = eps**2
        self.cert_thresh = cert_thresh

    def forward(self, warp: torch.Tensor, cert: torch.Tensor) -> torch.Tensor:
        """
         warp (Tensor): Estimated warp, shape (H, 2W, 4).
         cert (Tensor): Confidence values, shape (H, 2W, C).   — already a probability ∈ [0,1]
        """

        H, W, C = warp.shape
        w = W // 2

        warp_A2B, warp_B2A = torch.split(warp, w, dim=1)      # width split
        cert_A2B, cert_B2A = torch.split(cert, w, dim=1)

        crit = torch.nn.L1Loss()

        # optional certainty threshold
        if self.cert_thresh is not None:
            cert_mask_A2B = (cert_A2B >= self.cert_thresh).float()
            cert_mask_B2A = (cert_B2A >= self.cert_thresh).float()
            cert_A2B = cert_A2B * cert_mask_A2B
            cert_B2A = cert_B2A * cert_mask_B2A

        # aggregate
        L_warp = crit(warp[:, :,2:4], warp[:, :, 0:2])
        L_A2B = crit(warp_A2B[:, :,2:4], warp_A2B[:, :, 0:2])
        L_B2A = crit(warp_B2A[:, :,2:4], warp_B2A[:, :, 0:2])

        # return 0.5 * (L_A2B + L_B2A)
        return L_A2B

def lambda_warp(step, args):
    return 1.0
    # if step < args.plateau_step:                # linear warm-up
    #     return args.lambda_warp_plateau * step / args.plateau_step
    # if step < args.final_steps:               # flat plateau
    #     return args.lambda_warp_plateau
    # # cosine raise (or keep) toward lw_final
    # t  = (step - args.final_steps) / (args.max_train_steps - args.final_steps)
    # return args.lambda_warp_plateau + (args.lambda_warp_max - args.lambda_warp_plateau) * 0.5 * (1 - math.cos(math.pi * t))


