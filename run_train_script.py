#!/usr/bin/env python3
import subprocess
import shlex

def main():
    cmd = """
    accelerate launch train_contrastive_sdxl.py \
      --pretrained_model_name_or_path="stabilityai/sdxl-turbo" \
      --output_dir="output/day2night" \
      --dataset_folder="data/ECP" \
      --train_img_prep="sdxl_random_crop_512" \
      --val_img_prep="no_resize" \
      --learning_rate=1e-5 \
      --max_train_steps=12400 \
      --train_batch_size=1 \
      --gradient_accumulation_steps=1 \
      --lambda_gan=0.5 \
      --lambda_HDCE=1.0 \
      --lambda_SRC=1.0 \
      --dce_idt \
      --enable_xformers_memory_efficient_attention \
      --gradient_checkpointing \
      --report_to="wandb" \
      --tracker_project_name="day2night-baseline-debug" \
      --validation_steps=250
    """
    # shlex.split will turn that into a proper list of args
    args = shlex.split(cmd)
    # run it (and raise if it fails)
    subprocess.run(args, check=True)

if __name__ == "__main__":
    main()