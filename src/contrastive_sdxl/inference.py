from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
from diffusers.utils.import_utils import is_xformers_available

from .arguments import parse_args_inference

from .data.transforms import preprocess_inference
from .model import Contrastive_SDXL




def list_images(root, recursive=True):
    root = Path(root)
    paths = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        path for path in paths
        if path.is_file()
        and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )


def run_inference(args):

    if not args.model_path.is_file():
        raise ValueError("--model_path must point to one checkpoint file.")
    
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select --device cpu for CPU inference.")
    if args.use_fp16 and device.type != "cuda":
        raise ValueError("--use_fp16 requires CUDA.")
    
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.input_image is not None:
        if not args.input_image.is_file():
            raise FileNotFoundError(args.input_image)
        img_paths, input_root = [args.input_image], args.input_image.parent
    else:
        input_root = args.input_dir
        if not input_root.is_dir():
            raise NotADirectoryError(input_root)
        img_paths = list_images(input_root, not args.no_recursive)
    if not img_paths:
        raise ValueError("No input images found.")
    
    generated_dir = args.output_dir 
    output_paths = [generated_dir / path.relative_to(input_root).with_suffix(".png") for path in img_paths]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("Input filenames collide after conversion to PNG.")
    if any(src.resolve() == dst.resolve() for src, dst in zip(img_paths, output_paths)):
        raise ValueError("Generated files must not overwrite input images.")
    
    # for name in ("checkpoints", "generated", "evaluations", "logs"):
    #     (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.use_fp16 else torch.float32

    model = Contrastive_SDXL(args.model_path, device, dtype, args.pretrained_model_name_or_path, args.prompt)
    model.eval()

    if is_xformers_available() and device.type == "cuda":
        try:
            model.unet.enable_xformers_memory_efficient_attention()
        except (ValueError, RuntimeError) as error:
            print(f"Using standard attention: {error}")

    with torch.no_grad():
        for img_path, out_path in tqdm(list(zip(img_paths, output_paths)), desc="Translating", unit="image"):
            with Image.open(img_path) as image:
                img = image.convert("RGB")

            x_t, time_ids = preprocess_inference(img, args.image_prep, device, dtype)
            if any(size % 8 for size in x_t.shape[-2:]):
                raise ValueError(f"{img_path}: processed dimensions must be divisible by 8; use --image_prep resize_512.")
            
            out = model(x_t, caption=args.prompt, time_ids=time_ids)

            out_pil = F.to_pil_image(out[0].float().cpu() * 0.5 + 0.5)
            out_pil = out_pil.resize(img.size, Image.Resampling.LANCZOS)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_pil.save(out_path)
            
    return output_paths


def main(argv=None):
    args = parse_args_inference(argv)
    run_inference(args)


if __name__ == "__main__":
    main()
