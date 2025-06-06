import os
import random
import argparse
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as F
from glob import glob


def parse_args_unpaired_training():
    """
    Parses command-line arguments used for configuring an unpaired session (ContrastiveSDXL-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """

    parser = argparse.ArgumentParser(description="Contrastive SDXL")

    # fixed random seed
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # training args
    parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                        help='how to downsample the feature map')
    parser.add_argument('--netF_nc', type=int, default=256)
    parser.add_argument('--init_type', type=str, default='xavier',
                        choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--num_patches', type=int, default=128, help='number of patches per layer')
    parser.add_argument('--nce_layers', type=str, default='2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32',
                        help='compute NCE loss on which layers')  
    parser.add_argument('--nce_includes_all_negatives_from_minibatch', type=bool, nargs='?', const=True, default=False,
                        help='(used for single image translation) If True, include the negatives from the other samples of the minibatch.')

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)  # for CUT paper code, it is 1.0
    
    parser.add_argument("--lambda_idt_lpips", default=1.0, type=float)  # TEST
    parser.add_argument("--lambda_idt", default=1.0, type=float)  # TEST

    parser.add_argument('--lambda_HDCE', type=float, default=1.0, help='weight for HDCE loss')
    parser.add_argument('--lambda_SRC', type=float, default=1.0, help='weight for SRC loss')
    parser.add_argument('--dce_idt', action='store_true')

    parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')

    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--use_curriculum', action='store_true')
    parser.add_argument('--HDCE_gamma', type=float, default=1)
    parser.add_argument('--HDCE_gamma_min', type=float, default=1)
    parser.add_argument('--step_gamma', action='store_true')
    parser.add_argument('--step_gamma_epoch', type=int, default=200)
    parser.add_argument('--no_Hneg', action='store_true')

    # args for dataset and dataloader options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_img_prep", required=True)
    parser.add_argument("--val_img_prep", required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_epochs", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=None)

    # args for the model
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sdxl-turbo")
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--lora_rank_unet", default=64, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # args for validation and logging
    parser.add_argument("--viz_freq", type=int, default=20)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, required=True)
    parser.add_argument("--validation_steps", type=int, default=500, )
    parser.add_argument("--validation_num_images", type=int, default=-1,
                        help="Number of images to use for validation. -1 to use all images.")
    parser.add_argument("--checkpointing_steps", type=int, default=500)

    # args for the optimization options
    parser.add_argument("--learning_rate_netF", type=float, default=2e-4, help="The lr for netF.")
    parser.add_argument("--adam_beta1_netF", type=float, default=0.5,
                        help="The beta1 parameter for the Adam optimizer-netF.")
    parser.add_argument("--adam_beta2_netF", type=float, default=0.999,
                        help="The beta2 parameter for the Adam optimizer-netF.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'), )
    parser.add_argument("--lr_scheduler_netF", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.", )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # memory saving options
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to alloF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="Whether or not to use xformers.")

    args = parser.parse_args()
    return args


class RandomCropWithCoords:
    """
    A transform that performs a random crop on a PIL image and returns both 
    the cropped image and the coordinates of the crop's top-left corner.
    This is useful when you need to track exactly where the image was cropped
    (e.g., for SDXL's “text_time” micro-conditioning) while still using
    torchvision's RandomCrop logic under the hood.

    Attributes:
        size (int): The height and width of the square crop.
        interpolation (optional): Unused placeholder for API consistency;
            cropping operates by slicing, not resampling.
    """

    def __init__(self, size, interpolation=None):
        """
        Args:
            size (int): Desired output size of the crop (size x size).
            interpolation (optional): Ignored for cropping but kept for
                compatibility with other torchvision transforms.
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Apply a random crop to the input image and return crop plus coords.

        Args:
            img (PIL.Image.Image): Input image to be cropped.

        Returns:
            tuple:
                cropped (PIL.Image.Image): The randomly cropped sub-image of shape (size x size).
                coords (tuple[int, int]): The (x, y) coordinates of the top-left
                    corner of the crop in the original image's coordinate space.
        """
        # Sample random crop parameters (i: top, j: left, h: height, w: width)
        i, j, h, w = RandomCrop.get_params(img, output_size=(self.size, self.size))

        cropped = F.crop(img, i, j, h, w)

        # Return cropped image and (x=j, y=i) so downstream code can build time_ids
        return cropped, (j, i)


def build_transform(image_prep):
    """
    Builds a transformation pipeline tailored for SDXL-Turbo training and inference.
    This function provides multiple image preprocessing options based on the image_prep string.

    Args:
        image_prep (str): The preprocessing strategy name.

    Returns:
        torchvision.transforms.Compose or Lambda: The composed transform pipeline.
    """

    if image_prep == "sdxl_center_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(512)
        ])

    elif image_prep == "sdxl_random_crop_512":
        T = None  # Handle transformations manually in UnpairedDataset.__getitem__()

    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)
        ])

    elif image_prep == "no_resize":
        # For inference — skip any resizing/cropping
        T = transforms.Lambda(lambda x: x)

    else:
        raise ValueError(f"Unknown image_prep mode: {image_prep}")

    return T


class UnpairedDataset(torch.utils.data.Dataset):
    """
    Unpaired dataset loader for image-to-image translation (e.g., Contrastive SDXL-Turbo).

    This dataset loads unaligned images from two domains (A and B) and returns:
      - Processed source and target images
      - Corresponding fixed text prompts for each domain (raw strings, not tokenized)

    Captions are tokenized later inside the model to accommodate SDXL's dual-tokenizer system.
    """

    def __init__(self, dataset_folder, split, image_prep):
        """
        Args:
            dataset_folder (str): Root dataset directory containing subfolders (e.g., train_A, train_B).
            split (str): One of ['train', 'test'] — determines folder names to load.
            image_prep (str): Preprocessing pipeline string (e.g., 'sdxl_center_crop_512').
        """
        super().__init__()

        # Set source/target folders based on split
        if split == "train":
            self.source_folder = os.path.join(dataset_folder, "train_A")
            self.target_folder = os.path.join(dataset_folder, "train_B")
        elif split == "test":
            self.source_folder = os.path.join(dataset_folder, "test_A")
            self.target_folder = os.path.join(dataset_folder, "test_B")
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load fixed textual prompts
        with open(os.path.join(dataset_folder, "fixed_prompt_a.txt"), "r") as f:
            self.fixed_caption_src = f.read().strip()

        with open(os.path.join(dataset_folder, "fixed_prompt_b.txt"), "r") as f:
            self.fixed_caption_tgt = f.read().strip()

        # Get list of image files from both domains
        self.l_imgs_src = []
        self.l_imgs_tgt = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_src.extend(glob(os.path.join(self.source_folder, ext)))
            self.l_imgs_tgt.extend(glob(os.path.join(self.target_folder, ext)))

        self.image_prep = image_prep
        self.T = build_transform(image_prep)

    def __len__(self):
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def __getitem__(self, index):
        """
        Fetches a pair of unaligned images from domain A and B,
        along with their fixed raw captions.

        Returns:
            dict with keys:
              - 'pixel_values_src': preprocessed image tensor (3×H×W), float32, [-1, 1]
              - 'pixel_values_tgt': same as above
              - 'caption_src': str prompt for source image
              - 'caption_tgt': str prompt for target image
        """
        # Choose source image deterministically, fallback to random if out of bounds
        if index < len(self.l_imgs_src):
            img_path_src = self.l_imgs_src[index]
        else:
            img_path_src = random.choice(self.l_imgs_src)

        # Target image always sampled randomly
        img_path_tgt = random.choice(self.l_imgs_tgt)

        # Load and transform images
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")

        # Original dimensions
        h_original_src, w_original_src = img_pil_src.size[1], img_pil_src.size[0]
        h_original_tgt, w_original_tgt = img_pil_tgt.size[1], img_pil_tgt.size[0]

        if self.image_prep == "sdxl_random_crop_512":
            # Resize
            resize_size = int(512 * 1.25)
            img_pil_src = F.resize(img_pil_src, resize_size, interpolation=transforms.InterpolationMode.BICUBIC)
            img_pil_tgt = F.resize(img_pil_tgt, resize_size, interpolation=transforms.InterpolationMode.BICUBIC)

            # Get crop parameters
            i_src, j_src, h_crop, w_crop = transforms.RandomCrop.get_params(img_pil_src, output_size=(512, 512))
            i_tgt, j_tgt, _, _ = transforms.RandomCrop.get_params(img_pil_tgt, output_size=(512, 512))

            # Crop
            img_pil_src = F.crop(img_pil_src, i_src, j_src, h_crop, w_crop)
            img_pil_tgt = F.crop(img_pil_tgt, i_tgt, j_tgt, h_crop, w_crop)

            # Random Horizontal Flip
            if random.random() < 0.5:
                img_pil_src = F.hflip(img_pil_src)
            if random.random() < 0.5:
                img_pil_tgt = F.hflip(img_pil_tgt)

            # Construct time_ids
            time_ids_src = [h_original_src, w_original_src, i_src, j_src, h_crop, w_crop]
            time_ids_tgt = [h_original_tgt, w_original_tgt, i_tgt, j_tgt, h_crop, w_crop]
        else:
            # Apply predefined transform
            img_pil_src = self.T(img_pil_src)
            img_pil_tgt = self.T(img_pil_tgt)

            if self.image_prep == "sdxl_center_crop_512":
                time_ids_tgt = [h_original_tgt, w_original_tgt, 64, 344, img_pil_tgt.size[1], img_pil_tgt.size[0]]
                time_ids_src = [h_original_src, w_original_src, 64, 344, img_pil_src.size[1], img_pil_src.size[0]]
            else:
                time_ids_src = [h_original_src, w_original_src, 0, 0, img_pil_src.size[1], img_pil_src.size[0]]
                time_ids_tgt = [h_original_tgt, w_original_tgt, 0, 0, img_pil_tgt.size[1], img_pil_tgt.size[0]]

        # Normalize to [-1, 1]
        img_t_src = F.normalize(F.to_tensor(img_pil_src), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img_t_tgt = F.normalize(F.to_tensor(img_pil_tgt), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "time_ids_src": time_ids_src,
            "time_ids_tgt": time_ids_tgt,
        }

def data_dependent_initialize(
        data: torch.Tensor,
        netF: nn.Module,
        unet: nn.Module,
        vae_enc: nn.Module,
        attn_hook,
        nce_layers: list[int],
        args,
        accelerator,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        time_ids: torch.Tensor
):
    """
    Initializes netF using a single forward pass through UNet under the attention hook,
    on just one example.
    """

    # pick only one example (B=1)
    sample = data.to(accelerator.device)

    # no grad + mixed precision
    with torch.no_grad():
        # with accelerator.autocast():
        latent = vae_enc(sample)

        # run UNet+hook to grab attention maps
        features = attn_hook.extract_attention_features(latent=latent, unet=unet, timestep=timestep,
                                                        text_embed=prompt_embeds, pooled_embed=pooled_embeds,
                                                        time_ids=time_ids,
                                                        nce_layers=nce_layers)

        # this will trigger netF.create_mlp(...)
        _ = netF(features, num_patches=args.num_patches)

    # define & prepare optimizer for netF
    optimizer_F = torch.optim.AdamW(
        netF.parameters(),
        lr=args.learning_rate_netF,
        betas=(args.adam_beta1_netF, args.adam_beta2_netF),
    )
    optimizer_F = accelerator.prepare(optimizer_F)

    # clear any dangling features
    attn_hook.clear()

    return optimizer_F
