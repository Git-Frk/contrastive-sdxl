from pathlib import Path

import torch


class LabelBank:
    """YOLO labels normalized to the original image dimensions, indexed by stem.
    Each entry is a (N,5) float32 tensor: [cls, cx, cy, w, h] normalized to (W0,H0).
    """

    def __init__(self, labels_dir, device="cpu"):
        self.labels_dir = Path(labels_dir)
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Label directory does not exist: {self.labels_dir}")
        self.device = torch.device(device)
        self.bank = {}
        for path in self.labels_dir.glob("*.txt"):
            rows = []
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                values = line.split()
                if len(values) < 5:
                    raise ValueError(f"Invalid YOLO label in {path}: {line}")
                rows.append([float(values[0]), float(values[1]), float(values[2]),
                             float(values[3]), float(values[4])])
            self.bank[path.stem] = torch.tensor(rows, dtype=torch.float32, device=self.device).reshape(-1, 5)

    def get(self, stem):
        """Return (N,5) tensor for this stem; (0,5) if missing/empty."""
        if stem not in self.bank:
            raise FileNotFoundError(f"Missing guidance label: {self.labels_dir / (stem + '.txt')}")
        return self.bank[stem].clone()


@torch.no_grad()
def re_adjust_labels_to_random_crop512(labels_yolo_norm_orig, orig_hw, crop_i, crop_j,
                                     flip, resize_short=640, stem=None):
    """    
    Pipeline: unnormalize to original pixels -> resize(short=resize_short) -> crop(i,j,512)
              -> optional horizontal flip -> normalize to 512x512.
    Returns:
        (M,5) [cls, cx, cy, w, h] normalized to the 512x512 crop (after augs).
    """

    if labels_yolo_norm_orig is None or labels_yolo_norm_orig.numel() == 0:
        return None
    
    device = labels_yolo_norm_orig.device

    H0, W0 = orig_hw

    cls = labels_yolo_norm_orig[:, :1]

    # Unnormalize to original pixels 
    cx, cy, w, h = labels_yolo_norm_orig[:, 1:].unbind(-1)
    cx, cy, w, h = cx * W0, cy * H0, w * W0, h * H0   

    s = resize_short / float(min(H0, W0))

    # Resize to short side = resize_short, then crop [crop_j, crop_j+512] × [crop_i, crop_i+512]
    x1 = ((cx - 0.5 * w) * s - crop_j).clamp(0, 512)
    y1 = ((cy - 0.5 * h) * s - crop_i).clamp(0, 512)
    x2 = ((cx + 0.5 * w) * s - crop_j).clamp(0, 512)
    y2 = ((cy + 0.5 * h) * s - crop_i).clamp(0, 512)

    w2 = (x2 - x1).clamp(min=0)
    h2 = (y2 - y1).clamp(min=0)

    keep = (w2 >= 1) & (h2 >= 1)
    if not keep.any():
        return None
    
    x1, y1, w2, h2, cls = x1[keep], y1[keep], w2[keep], h2[keep], cls[keep]

    if flip:
        x1 = 512 - (x1 + w2)

    # Back to normalized [cx,cy,w,h] in 512×512
    boxes = torch.stack([(x1 + 0.5 * w2) / 512, (y1 + 0.5 * h2) / 512, w2 / 512, h2 / 512], -1)

    return torch.cat([cls, boxes], -1).to(device)
