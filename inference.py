import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from constrastive_sdxl import ContrastiveSDXL_Turbo
from utils.train_utils import build_transform
from diffusers.utils.import_utils import is_xformers_available

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='path to the input image')
    parser.add_argument('--prompt', type=str, required=False,
                        help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument('--model_name', type=str, default=None, help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='no_resize', help='the image preparation method')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name is None != args.model_path is None:
        raise ValueError('Either model_name or model_path should be provided')

    if args.model_path is not None and args.prompt is None:
        # raise ValueError('prompt is required when loading a custom model_path.')
        # args.prompt = ("Driving at night. Artificial lights are accurately positioned: streetlights cast a glow on sidewalks, "
        #                "car headlights illuminate the road with focused beams, and building windows emit a soft interior light. "
        #                "Shadows and reflections follow realworld physics, with ambient darkness ensuring that all illumination comes from natural sources.")
        args.prompt = ("Driving at night.")

    if args.model_name is not None:
        assert args.prompt is None, 'prompt is not required when loading a pretrained model.'
        assert args.direction is None, 'direction is not required when loading a pretrained model.'

    # initialize the model
    model = ContrastiveSDXL_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.eval()
    
    if is_xformers_available():
        try:
            model.unet.enable_xformers_memory_efficient_attention()
            print("🚀 xFormers memory-efficient attention enabled")
        except Exception as e:
            print(f"⚠️  Could not enable xFormers attention: {e}")
    else:
        print("ℹ️  xFormers not installed; using standard attention")

        if args.use_fp16:
            model.half()

    T_val = build_transform(args.image_prep)

    input_image = Image.open(args.input_image).convert('RGB')
    time_ids = [input_image.size[1], input_image.size[0], 0, 0, input_image.size[1], input_image.size[0]]

    # translate the image
    with torch.no_grad():
        input_img = T_val(input_image)
        x_t = transforms.ToTensor()(input_img)
        x_t = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(x_t).unsqueeze(0).cuda()
        time_ids = torch.tensor(time_ids, device=x_t.device)
        if args.use_fp16:
            x_t = x_t.half()
        output = model(x_t, caption=args.prompt, time_ids = time_ids)

    output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
    output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)  # Try LANCZOS,BICUBIC too

    # save the output image
    # bname = os.path.basename(args.input_image)
    # os.makedirs(args.output_dir, exist_ok=True)
    # output_pil.save(os.path.join(args.output_dir, bname))

    output_pil.save(args.output_dir) # for generating multiple image using inference_day2night.py

