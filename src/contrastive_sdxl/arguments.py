import argparse
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def str_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args_unpaired_training(argv=None):
    parser = argparse.ArgumentParser(description="Train Contrastive-SDXL on unpaired images.")
    parser.add_argument("--dataset_folder", default="data/ECP")
    parser.add_argument("--output_dir", default="outputs", help="Root for checkpoints, generated images, evaluations, and logs.")
    parser.add_argument("--prompt", default="Driving at night.")
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sdxl-turbo")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--train_img_prep", choices=("sdxl_random_crop_512", "sdxl_center_crop_512", "resize_512", "resize_512x512", "no_resize"), default="sdxl_random_crop_512")
    parser.add_argument("--train_batch_size", type=positive_int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--max_train_epochs", type=positive_int, default=10)
    parser.add_argument("--max_train_steps", type=positive_int, default=15000)
    parser.add_argument("--gradient_accumulation_steps", type=positive_int, default=1)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--lora_rank_unet", type=positive_int, default=64)
    parser.add_argument("--lora_rank_vae", type=positive_int, default=4)
    parser.add_argument("--patch_projector", choices=("mlp_sample", "sample"), default="mlp_sample")
    parser.add_argument("--patch_projector_nc", type=positive_int, default=256)
    parser.add_argument("--init_type", choices=("normal", "xavier", "kaiming", "orthogonal"), default="xavier")
    parser.add_argument("--init_gain", type=float, default=0.02)
    parser.add_argument("--num_patches", type=positive_int, default=128, help='number of patches per layer')
    parser.add_argument("--nce_layers", default="30,32,34,36,38",  help='compute NCE loss on which layers') 
    parser.add_argument("--dino_repo", default=None, help="Optional local DINOv2 checkout for torch.hub.")
    parser.add_argument("--nce_T", type=float, default=0.07)
    parser.add_argument('--HDCE_gamma', type=float, default=1) #If using curriculum, the starting value of gamma will decay to HDCE_gamma_min over the course of training; therefore set it to a larger value than HDCE_gamma_min
    parser.add_argument("--HDCE_gamma_min", type=float, default=5)
    parser.add_argument("--use_curriculum", action="store_true") 
    parser.add_argument('--step_gamma', action='store_true')  # If True, gamma will be set to 1 after step_gamma_epoch
    parser.add_argument('--step_gamma_epoch', type=int, default=4)
    parser.add_argument("--no_Hneg", action="store_true")
    parser.add_argument("--gan_disc_type", choices=("vagan_clip",), default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", type=float, default=0.5)
    parser.add_argument("--lambda_HDCE", type=float, default=1)
    parser.add_argument("--lambda_SRC", type=float, default=1)
    parser.add_argument("--dce_idt", action="store_true")
    parser.add_argument("--lambda_idt", type=float, default=1)
    parser.add_argument("--lambda_idt_lpips", type=float, default=1)
    parser.add_argument("--det_loss", action="store_true")
    parser.add_argument("--lambda_det", type=float, default=1)
    parser.add_argument("--detector_weights", default="outputs/checkpoints/yolo11/best.pt")
    parser.add_argument("--labels_dir", default="data/ECP/labels/train_src")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--learning_rate_patch_projector", type=float, default=2e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_beta1_patch_projector", type=float, default=0.5)
    parser.add_argument("--adam_beta2_patch_projector", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=10)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_scheduler_patch_projector", default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=positive_int, default=1)
    parser.add_argument("--lr_power", type=float, default=1)
    parser.add_argument("--viz_freq", type=positive_int, default=20)
    parser.add_argument("--checkpointing_steps", type=positive_int, default=500)
    parser.add_argument("--report_to", choices=("wandb", "none"), default="wandb")
    parser.add_argument("--tracker_project_name", default="contrastive-sdxl")

    args = parser.parse_args(argv)
    
    if args.train_batch_size != 1:
        parser.error("The original SRC/hDCE training path requires --train_batch_size 1.")
    if args.gradient_accumulation_steps != 1:
        parser.error("The separate generator updates require --gradient_accumulation_steps 1.")
    if args.det_loss and args.train_img_prep != "sdxl_random_crop_512":
        parser.error("Detector guidance requires --train_img_prep sdxl_random_crop_512.")
    if args.dataloader_num_workers < 0 or args.lr_warmup_steps < 0:
        parser.error("Worker and warmup counts cannot be negative.")
    for name in ("learning_rate", "learning_rate_patch_projector", "nce_T", "HDCE_gamma", "HDCE_gamma_min", "max_grad_norm"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be greater than zero.")
    for name in ("lambda_gan", "lambda_HDCE", "lambda_SRC", "lambda_idt", "lambda_idt_lpips", "lambda_det"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} cannot be negative.")
    try:
        layers = [int(i.strip()) for i in args.nce_layers.split(",")]
    except ValueError:
        parser.error("--nce_layers must contain comma-separated integers.")
    if not layers or len(set(layers)) != len(layers) or any(i < 0 or i >= 40 for i in layers):
        parser.error("--nce_layers must contain distinct DINOv2 ViT-g block indices from 0 to 39.")
    return args


def parse_args_inference(argv=None):
    parser = argparse.ArgumentParser(description="Translate an image or folder using one SDXL checkpoint.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input_image", type=Path)
    inputs.add_argument("--input_dir", type=Path)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sdxl-turbo")
    parser.add_argument("--prompt", default="Driving at night.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--image_prep", choices=("no_resize", "resize_512", "resize_512x512", "sdxl_center_crop_512"), default="no_resize")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_fp16", "--fp16", action="store_true")
    parser.add_argument("--no_recursive", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)
