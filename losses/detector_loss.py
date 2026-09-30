from __future__ import annotations
from types import SimpleNamespace
from typing import List, Union, Dict, Tuple
from pathlib import Path
import torch
from torch import Tensor
from ultralytics import YOLO


def _to_namespace(x):
    """
    Recursively convert dicts to SimpleNamespace so attributes can be accessed with dots.
    """
    if isinstance(x, SimpleNamespace):
        return x
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in x.items()})
    return x


def _ensure_loss_hyperparameters(hyp: SimpleNamespace) -> SimpleNamespace:
    """
    Ensure v8/11 detection loss hyperparameters(box, cls, dfl, etc.) exist on the namespace. If not, it sets default values.
    """
    # Core detection gains
    if not hasattr(hyp, "box"): 
        hyp.box = 7.5        # box regression gain
    if not hasattr(hyp, "cls"): 
        hyp.cls = 0.5        # classification gain
    if not hasattr(hyp, "dfl"): 
        hyp.dfl = 1.5        # distribution focal loss gain

    # Focal / labeling knobs 
    if not hasattr(hyp, "fl_gamma"): hyp.fl_gamma = 0.0
    if not hasattr(hyp, "label_smoothing"): hyp.label_smoothing = 0.0

    # Occasionally referenced 
    if not hasattr(hyp, "obj"): hyp.obj = 1.0

    # Nominal batch size 
    if not hasattr(hyp, "nbs"): hyp.nbs = 64

    return hyp


# ------------------------ YOLO-guided loss --------------------- #

class YoloDetectionGuidance:
    """
    Compute YOLOv11 detection loss on generated images as an auxiliary guidance term.
      - Detector weights are frozen; gradients flow from loss -> detector ops -> x_fake -> generator.
    """

    def __init__(
        self,
        weights_or_model: Union[str, YOLO] = "yolo11s.pt",
        device: Union[str, torch.device] = "cuda:0",
    ):
        """
        Args:
            weights_or_model:
                - str: checkpoint name/path (e.g., "yolo11s.pt")
                - YOLO: an already-constructed Ultralytics wrapper
            device: compute device (e.g., "cuda:0", "cpu", torch.device(...))
        """
        self.device = torch.device(device) if isinstance(device, str) else device


        if isinstance(weights_or_model, YOLO):
            self.det = weights_or_model
        elif isinstance(weights_or_model, str):
            self.det = YOLO(weights_or_model)
        else:
            raise TypeError("weights_or_model must be a str checkpoint or a YOLO object.")

        self._finalize_detector()

    def _finalize_detector(self) -> None:
        """Freeze the detector, normalize args/hyperparameters, and prepare the official loss criterion."""
        # torch.nn.Module inside the YOLO wrapper
        self.net = self.det.model.to(self.device)

        # Only want gradients w.r.t. x_fake is required
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()  

        # Ensure the detect head is not in export mode (export=True alters output layout)
        try:
            detect = self.net.model[-1]
            if hasattr(detect, "export"):
                detect.export = False
        except Exception:
            pass

        # Normalize model.args to a dot-accessible structure 
        try:
            from omegaconf import DictConfig
            if isinstance(getattr(self.net, "args", None), DictConfig):
                self.net.args = _to_namespace(self.net.args.to_container(resolve=True))
        except Exception:
            pass
        if isinstance(getattr(self.net, "args", None), dict):
            self.net.args = _to_namespace(self.net.args)

        # Build the official Ultralytics loss and ensure its hyperparamter is complete
        try:
            from ultralytics.utils.loss import v8DetectionLoss
        except Exception:
            from ultralytics.yolo.utils.loss import v8DetectionLoss

        self.net.criterion = v8DetectionLoss(self.net)

        hyperpramters = getattr(self.net.criterion, "hyp", None)
        hyperpramter_namespace = _to_namespace(hyperpramters) if hyperpramters is not None else SimpleNamespace()
        self.net.criterion.hyp = _ensure_loss_hyperparameters(hyperpramter_namespace)

    # -------------------------- static utilities -------------------------- #

    @staticmethod
    def _cat_targets(labels_list: List[Tensor], device: Union[str, torch.device]) -> Tensor:
        """
        Concatenate per-image YOLO labels (Ni,5) -> (M,6) as [b, cls, cx, cy, w, h].

        Args:
            labels_list: length-B list; each element is (Ni,5) tensor with rows
                         [cls, cx, cy, w, h] normalized to the model's input frame.
            device: target device for the output tensor.

        Returns:
            (M,6) tensor if there are no objects).
        """
        pieces: List[Tensor] = []
        for b, lab in enumerate(labels_list):
            if lab is None or lab.numel() == 0:
                continue
            if lab.shape[-1] != 5:
                raise ValueError(
                    f"Label tensor for batch index {b} must have shape (Ni,5) "
                    f"[cls,cx,cy,w,h], got {tuple(lab.shape)}"
                )
            # prepend the image index
            bi = torch.full((lab.shape[0], 1), b, dtype=lab.dtype, device=lab.device)
            pieces.append(torch.cat([bi, lab], dim=1))

        if not pieces:
            return torch.zeros((0, 6), dtype=torch.float32, device=device)

        return torch.cat(pieces, dim=0).to(device, non_blocking=True)

    @staticmethod
    def _to_scalar_loss(out, device: Union[str, torch.device]) -> Tensor:
        """
        Normalize Ultralytics' return value into a scalar tensor suitable for backprop.
        """
        if isinstance(out, (tuple, list)):
            loss = out[0]
        else:
            loss = out

        if isinstance(loss, dict):
            # Prefer explicit aggregate keys if present; otherwise sum tensors
            loss = loss.get("loss") or loss.get("total") or sum(
                v for v in loss.values() if torch.is_tensor(v)
            )

        if not torch.is_tensor(loss):
            loss = torch.as_tensor(loss, dtype=torch.float32, device=device)

        # Reduce to scalar if needed
        if loss.ndim == 0:
            return loss
        if loss.ndim == 1 and loss.numel() in (3, 4):  # [box, cls, dfl, (obj)]
            return loss.sum()
        return loss.mean()

    # ------------------------------ forward ------------------------------- #

    @torch.enable_grad()
    def __call__(self, x_fake: Tensor, labels_list: List[Tensor]) -> Tensor:
        """
        Compute the YOLO detection loss on generated images.

        Args:
            x_fake:  (B, 3, H, W) float tensor. If your generator outputs in [-1,1],
                     pass it directly; we remap to [0,1] here. Expected geometry must
                     match your label normalization.
            labels_list: list of length B; each element is a (Ni,5) tensor with rows
                         [cls, cx, cy, w, h], normalized to the SAME frame as x_fake.

        Returns:
            Scalar tensor (requires_grad=True) suitable for backprop through the generator.
        """
        if x_fake.ndim != 4 or x_fake.shape[1] != 3:
            raise ValueError(f"x_fake must be (B,3,H,W), got {tuple(x_fake.shape)}")

        # Map from [-1,1] (common for diffusion) to [0,1] for YOLO
        x_fake = (x_fake.clamp(-1, 1) + 1.0) * 0.5
        x_fake = x_fake.to(self.device, dtype=torch.float32, non_blocking=True).contiguous()

        # Pack targets for Ultralytics' training-loss path
        targets = self._cat_targets(labels_list, device=self.device)  # (M,6) or (0,6)
        batch = {
            "img": x_fake,                        # (B,3,H,W) in [0,1]
            "batch_idx": targets[:, 0].long(),    # (M,)
            "cls": targets[:, 1:2],               # (M,1)
            "bboxes": targets[:, 2:6],            # (M,4) normalized cx,cy,w,h
        }

        # call the inner torch module; with a dict, it returns training loss
        out = self.net(batch) # returns loss 
        loss = self._to_scalar_loss(out, x_fake.device)

        if not loss.requires_grad:
            raise RuntimeError("Detection loss does not require grad; check graph construction.")

        return loss
    


class LabelBank:
    """
    Loads YOLO .txt labels normalized to ORIGINAL image size into memory once.
    Access by image stem (filename without extension).

    Each entry is a (N,5) float32 tensor: [cls, cx, cy, w, h] normalized to (W0,H0).
    """

    def __init__(self, labels_dir: Path, device="cpu"):
        self.labels_dir = Path(labels_dir)
        self.device = torch.device(device)
        self.bank: Dict[str, torch.Tensor] = {}
        self._load_all()

    def _load_all(self):
        for p in self.labels_dir.glob("*.txt"):
            stem = p.stem
            rows = []
            txt = p.read_text().strip()
            if txt:
                for ln in txt.splitlines():
                    parts = ln.strip().split()
                    if len(parts) >= 5:
                        rows.append([float(parts[0]), float(parts[1]), float(parts[2]),
                                     float(parts[3]), float(parts[4])])
            if rows:
                t = torch.tensor(rows, dtype=torch.float32, device=self.device)
            else:
                t = torch.zeros((0, 5), dtype=torch.float32, device=self.device)
            self.bank[stem] = t

    def get(self, stem: str) -> torch.Tensor:
        """Return (N,5) tensor for this stem; (0,5) if missing/empty."""
        try:
            return self.bank[stem].clone()
        except Exception:
            print(f"LabelBank: cannot find {stem} in {self.labels_dir}\n")


@torch.no_grad()
def re_adjust_labels_to_random_crop512(
    labels_yolo_norm_orig: torch.Tensor,  # (N,5) [cls,cx,cy,w,h] normalized to ORIGINAL (W0,H0)
    orig_hw: Tuple[int, int],             # (H0, W0)
    crop_i: int, crop_j: int,             # top/left of the 512 crop (in resized image coords)
    flip: bool,
    resize_short: int = 640,              # your 512 * 1.25
    stem: str = None                        # for debug prints only
) -> torch.Tensor:
    """
    Pipeline: unnormalize to ORIGINAL pixels -> resize(short=resize_short) -> crop(i,j,512)
              -> optional horizontal flip -> normalize to 512x512.

    Returns:
        (M,5) [cls, cx, cy, w, h] normalized to the 512x512 crop (after augs).
    """
    device = labels_yolo_norm_orig.device
    if labels_yolo_norm_orig is None or labels_yolo_norm_orig.numel() == 0:
        return None

    H0, W0 = orig_hw
    
    mx = labels_yolo_norm_orig[:,1:].max().item()
    if mx > 1.0:
        print(f"[DEBUG] {stem}: max_label={mx:.3f} (should be <=1), H0={H0} W0={W0} i={crop_i} j={crop_j} flip={flip} rs={resize_short}")

    cls = labels_yolo_norm_orig[:, 0:1]
    cx, cy, w, h = labels_yolo_norm_orig[:, 1:].unbind(-1)

    # Unnormalize from original
    cx *= W0; cy *= H0; w *= W0; h *= H0
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    # Resize ORIGINAL -> resized where short side == resize_short
    s = resize_short / float(min(H0, W0))
    x1 *= s; y1 *= s; x2 *= s; y2 *= s

    # Crop window [crop_j, crop_j+512] × [crop_i, crop_i+512]
    x1 = (x1 - crop_j).clamp(0, 512)
    y1 = (y1 - crop_i).clamp(0, 512)
    x2 = (x2 - crop_j).clamp(0, 512)
    y2 = (y2 - crop_i).clamp(0, 512)

    w2 = (x2 - x1).clamp(min=0)
    h2 = (y2 - y1).clamp(min=0)

    keep = (w2 >= 1.0) & (h2 >= 1.0)  # drop boxes that vanish
    if keep.sum() == 0:
        return None

    x1, y1, w2, h2, cls = x1[keep], y1[keep], w2[keep], h2[keep], cls[keep]

    # Optional horizontal flip on the CROPPED 512×512 image
    if flip:
        x1 = 512.0 - (x1 + w2)  # reflect in 512-wide crop
        x2 = x1 + w2
    # Back to normalized cx,cy,w,h in 512×512
    cx = (x1 + 0.5 * w2) / 512.0
    cy = (y1 + 0.5 * h2) / 512.0
    ww = w2 / 512.0
    hh = h2 / 512.0

    return torch.cat([cls, torch.stack([cx, cy, ww, hh], dim=-1)], dim=-1).to(device)

