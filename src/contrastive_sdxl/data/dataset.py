import random
from glob import glob
from pathlib import Path

import torch
from PIL import Image

from .transforms import build_transform, normalize_image, random_crop_pair


class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split="train", image_prep="sdxl_random_crop_512"):
        if split not in ("train", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.source_folder = Path(dataset_folder) / f"{split}_src"
        self.target_folder = Path(dataset_folder) / f"{split}_tgt"
        self.l_imgs_src = self._images(self.source_folder)
        self.l_imgs_tgt = self._images(self.target_folder)
        for folder, paths in ((self.source_folder, self.l_imgs_src), (self.target_folder, self.l_imgs_tgt)):
            if not paths:
                raise ValueError(f"No images found in {folder}")
        self.image_prep = image_prep
        self.T = build_transform(image_prep)

    @staticmethod
    def _images(folder):
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"):
            paths.extend(glob(str(folder / ext)))
        return paths

    def __len__(self):
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def __getitem__(self, index):
        img_path_src = self.l_imgs_src[index] if index < len(self.l_imgs_src) else random.choice(self.l_imgs_src)
        img_path_tgt = random.choice(self.l_imgs_tgt)
        with Image.open(img_path_src) as img:
            img_pil_src = img.convert("RGB")
        with Image.open(img_path_tgt) as img:
            img_pil_tgt = img.convert("RGB")

        flip_src = flip_tgt = False
        if self.image_prep == "sdxl_random_crop_512":
            (img_pil_src, img_pil_tgt, time_ids_src, time_ids_tgt,
             flip_src, flip_tgt) = random_crop_pair(img_pil_src, img_pil_tgt)
        else:
            img_pil_src, time_ids_src = self._transform(img_pil_src)
            img_pil_tgt, time_ids_tgt = self._transform(img_pil_tgt)

        return {
            "pixel_values_src": normalize_image(img_pil_src),
            "pixel_values_tgt": normalize_image(img_pil_tgt),
            "time_ids_src": torch.tensor(time_ids_src),
            "time_ids_tgt": torch.tensor(time_ids_tgt),
            "flip_src": flip_src, "flip_tgt": flip_tgt,
            "stem_src": Path(img_path_src).stem, "stem_tgt": Path(img_path_tgt).stem,
            "path_src": img_path_src, "path_tgt": img_path_tgt,
        }

    def _transform(self, img):
        h_original, w_original = img.height, img.width
        img = self.T(img)
        # Preserve the center-crop conditioning used by the original dataset.
        i, j = (64, 344) if self.image_prep == "sdxl_center_crop_512" else (0, 0)
        return img, [h_original, w_original, i, j, img.height, img.width]
