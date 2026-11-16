import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F


IMAGE_PREPS = ("sdxl_random_crop_512", "sdxl_center_crop_512", "resize_512", "resize_512x512", "no_resize")


def build_transform(image_prep):
    if image_prep == "sdxl_random_crop_512":
        return None
    if image_prep == "sdxl_center_crop_512":
        return transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(512),
        ])
    if image_prep in ("resize_512", "resize_512x512"):
        return transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)
    if image_prep == "no_resize":
        return transforms.Lambda(lambda img: img)
    raise ValueError(f"Unknown image_prep: {image_prep}")


def normalize_image(img):
    img = F.to_tensor(img)
    img = F.normalize(img, [0.5] * 3, [0.5] * 3)  # F.normalize(img, mean, std) --> normalize to [-1, 1] 
    return img


def random_crop_pair(img_pil_src, img_pil_tgt):
    """Keep the original order of crop and flip random draws."""
    h_src, w_src = img_pil_src.height, img_pil_src.width
    h_tgt, w_tgt = img_pil_tgt.height, img_pil_tgt.width
    interpolation = transforms.InterpolationMode.BICUBIC
    img_pil_src = F.resize(img_pil_src, 640, interpolation=interpolation)
    img_pil_tgt = F.resize(img_pil_tgt, 640, interpolation=interpolation)
    i_src, j_src, h_crop, w_crop = transforms.RandomCrop.get_params(img_pil_src, (512, 512))
    i_tgt, j_tgt, _, _ = transforms.RandomCrop.get_params(img_pil_tgt, (512, 512))
    img_pil_src = F.crop(img_pil_src, i_src, j_src, h_crop, w_crop)
    img_pil_tgt = F.crop(img_pil_tgt, i_tgt, j_tgt, h_crop, w_crop)
    flip_src = random.random() < 0.5
    if flip_src:
        img_pil_src = F.hflip(img_pil_src)
    flip_tgt = random.random() < 0.5
    if flip_tgt:
        img_pil_tgt = F.hflip(img_pil_tgt)
    time_ids_src = [h_src, w_src, i_src, j_src, h_crop, w_crop]
    time_ids_tgt = [h_tgt, w_tgt, i_tgt, j_tgt, h_crop, w_crop]
    return img_pil_src, img_pil_tgt, time_ids_src, time_ids_tgt, flip_src, flip_tgt


def preprocess_inference(img, image_prep, device, dtype):
    T_val = build_transform(image_prep)
    if T_val is None:
        raise ValueError("Random cropping is only supported during training.")

    time_ids = torch.tensor([[img.height, img.width, 0, 0, img.height, img.width]], device=device)

    img = T_val(img)
    x_t = normalize_image(img).unsqueeze(0).to(device=device, dtype=dtype)
    
    return x_t, time_ids
