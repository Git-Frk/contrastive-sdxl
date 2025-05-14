# contrastive-sdxl

This work implements a novel training setup for unpaired translation tasks (e.g., day-to-night) by combining a powerful SDXL-Turbo backbone with lightweight fine-tuning and contrastive learning techniques.
## 🚀 Overview

This project builds on top of **SDXL-Turbo** and:

- **Patch-wise contrastive losses**: including Semantic Relation Consistency (SRC) and hierarchical Dense Contrastive Estimation (hDCE).
- **LoRA-based fine-tuning** of UNet and VAE modules for memory-efficient training.
- **Self-attention feature extraction** using hooks for computing contrastive losses.
- **GAN loss integration** for improving visual realism.

## 📁 Project Structure

```
contrastive-sdxl/
│
├── train_contrastive_sdxl.py       # Main training script
├── run_day2night.py                # Sample launcher script
│
├── constrastive_sdxl.py            # Inference and helper functions
├── utils/
│   └── train_utils.py              # Utilities for training and initialization
│
├── models/
│   ├── patch_network.py            # PatchSampleF contrastive head
│   ├── unet/
│   │   ├── unet_attention_hook.py  # Hooks to extract attention features
│   │   └── unet_factory.py         # LoRA initialization for SDXL UNet
│   └── vae/
│       └── custom_vae_forward.py   # LoRA+skip-injected VAE forward functions
│
├── losses/
│   ├── SRC.py                      # Semantic Relation Consistency Loss
│   └── hDCE.py                     # Patch-wise Contrastive Loss
│
└── requirements.txt                # Dependencies
```

## 🔧 Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/Git-Frk/contrastive-sdxl.git
   cd contrastive-sdxl
   ```

2. Create a Conda environment:
   ```bash
   conda create -n contrastive-sdxl python=3.10
   conda activate contrastive-sdxl
   pip install -r requirements.txt
   ```

3. Configure `accelerate`:
   ```bash
   accelerate config
   ```

## 🏁 Training

To launch training:
```bash
  accelerate launch train_contrastive_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/sdxl-turbo" \
    --output_dir="output/day2night" \
    --dataset_folder="data/ECP" \
    --train_img_prep="sdxl_random_crop_512" \
    --val_img_prep="no_resize" \
    --learning_rate=1e-5 \
    --max_train_steps=25000 \
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
```

## 🧪 Features

- ✅ SDXL-Turbo support with dual text encoders
- ✅ Patch-wise contrastive learning (SRC + hDCE)
- ✅ LoRA on UNet + VAE + skip-conv modules
- ✅ Attention hook-based feature extraction

## 🧠 Citation

If you find this repo useful, please consider citing relevant works:

To Do

## 📜 License

To Do
