from types import SimpleNamespace
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
# from ultralytics.yolo.utils.loss import v8DetectionLoss
import torch


class YoloDetectionGuidance:
    """Frozen YOLO detector with gradients through its input image."""

    def __init__(self, weights_or_model, device="cuda"):
        self.device = torch.device(device)

        if isinstance(weights_or_model, YOLO):
            self.det = weights_or_model
        elif isinstance(weights_or_model, str):
            self.det = YOLO(weights_or_model)
        else:
            raise TypeError("weights_or_model must be a str checkpoint or a YOLO object.")
        
        self.net = self.det.model.to(self.device)
        self.net.requires_grad_(False)
        self.net.eval()
        self.net.model[-1].export = False # Ensure the detect head is not in export mode (export=True alters output layout)
        if isinstance(self.net.args, dict):
            self.net.args = SimpleNamespace(**self.net.args)
        for name, value in {"box": 7.5, "cls": 0.5, "dfl": 1.5, "fl_gamma": 0.0,
                            "label_smoothing": 0.0, "obj": 1.0, "nbs": 64}.items():
            if not hasattr(self.net.args, name):
                setattr(self.net.args, name, value)
        self.net.criterion = v8DetectionLoss(self.net)

    @staticmethod
    def _cat_targets(labels_list, device):
        """
        Concatenate per-image YOLO labels (Ni,5) -> (M,6) as [b, cls, cx, cy, w, h].
        Args:
            labels_list: length-B list; each element is (N,5) tensor with rows
                         [cls, cx, cy, w, h] normalized to the model's input frame.
            device: target device for the output tensor.
        Returns:
            (M,6) tensor if there are no objects).
        """

        pieces = []
        for b, lab in enumerate(labels_list):
            if lab is None or lab.numel() == 0:
                continue
            if lab.ndim != 2 or lab.shape[1] != 5:
                raise ValueError("Each label tensor must have shape (N, 5)")

            # prepend the image index
            bi = torch.full((lab.shape[0], 1), b, device=lab.device, dtype=lab.dtype)
            pieces.append(torch.cat([bi, lab], dim=1))

        if not pieces:
            return torch.zeros((0, 6), dtype=torch.float32, device=device)

        return torch.cat(pieces, dim=0).to(device, non_blocking=True)

    @staticmethod
    def _to_scalar_loss(out, device):
        """
        Normalize Ultralytics' return value into a scalar tensor suitable for backprop.
        """

        if isinstance(out, (tuple, list)):
            loss = out[0]
        else:
            loss = out

        if isinstance(loss, dict):
            if "loss" in loss:
                loss = loss["loss"]
            elif "total" in loss:
                loss = loss["total"]
            else:
                loss = sum(value for value in loss.values() if torch.is_tensor(value))

        if not torch.is_tensor(loss):
            loss = torch.as_tensor(loss, dtype=torch.float32, device=device)

        if loss.ndim == 0:
            return loss
        if loss.ndim == 1 and loss.numel() in (3, 4):  # [box, cls, dfl, (obj)]
            return loss.sum()
        return loss.mean()

    @torch.enable_grad()
    def __call__(self, x_fake, labels_list):
        """
        Compute the YOLO detection loss on generated images.
        Args:
            x_fake:  (B, 3, H, W) float tensor. If the generator outputs is in [-1,1],
                     pass it directly; the call remaps to [0,1]. 
            labels_list: list of length B; each element is a (Ni,5) tensor with rows
                         [cls, cx, cy, w, h], normalized to the SAME frame as x_fake.
        Returns:
            Scalar tensor (requires_grad=True) suitable for backprop through the generator.
        """

        if x_fake.ndim != 4 or x_fake.shape[1] != 3 or len(labels_list) != len(x_fake):
            raise ValueError("Expected (B, 3, H, W) images and one label entry per image")
        
        # Map from [-1,1] (common for diffusion) to [0,1] for YOLO
        x_fake = (x_fake.clamp(-1, 1) + 1) * 0.5
        x_fake = x_fake.to(self.device, dtype=torch.float32, non_blocking=True).contiguous()

        targets = self._cat_targets(labels_list, device=self.device) # (M,6) or (0,6)

        batch = {"img": x_fake,                         # (B,3,H,W) in [0,1]
                 "batch_idx": targets[:, 0].long(),     # (M,)
                 "cls": targets[:, 1:2],                # (M,1)
                 "bboxes": targets[:, 2:6]}             # (M,4) normalized cx,cy,w,h
        
        out = self.net(batch)
        loss = self._to_scalar_loss(out, x_fake.device)
        
        if not loss.requires_grad:
            raise RuntimeError("Detector loss is disconnected from the generated image")
        return loss
