import os, torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from tqdm import tqdm
from lpips import LPIPS
from diffusers import AutoencoderKL
import wandb
from PIL import Image
import glob, itertools, random

class FlatFolderDataset(torch.utils.data.Dataset):
    """Loads every image directly inside `root` (no subfolders)."""
    def __init__(self, root, label, transform=None,
                 exts=("jpg","jpeg","png","bmp","webp")):
        self.paths = list(itertools.chain.from_iterable(
            glob.glob(os.path.join(root, f"*.{ext}")) for ext in exts))
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")
        self.label = label        # 0 = day, 1 = night
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.label

# -------------------- VAE patch --------------------
def patch_vae_for_skip(vae: AutoencoderKL):
    # encoder forward --------------------------------------------------------
    def encoder_fwd(self, x):
        x0 = self.conv_in(x)
        skips = []
        feat = x0
        for blk in self.down_blocks:
            skips.append(feat)
            feat = blk(feat)
        feat = self.mid_block(feat)
        feat = self.conv_norm_out(feat)
        feat = self.conv_act(feat)
        feat = self.conv_out(feat)
        self.current_down_blocks = skips
        return feat
    vae.encoder.forward = encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)

    # zero‑conv skip layers ---------------------------------------------------
    def zconv(cin, cout):
        m = nn.Conv2d(cin, cout, 1, bias=False, device=device, dtype=dtype)
        nn.init.constant_(m.weight, 1e-5)
        return m
    vae.decoder.skip_conv_1 = zconv(512, 512)
    vae.decoder.skip_conv_2 = zconv(256, 512)
    vae.decoder.skip_conv_3 = zconv(128, 512)
    vae.decoder.skip_conv_4 = zconv(128, 256)

    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1.0

    # decoder forward ---------------------------------------------------------
    def decoder_fwd(self, sample, latent_embeds=None):
        sample = self.conv_in(sample)
        sample = self.mid_block(sample, latent_embeds)

        skips = self.incoming_skip_acts[::-1]
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = getattr(self, f"skip_conv_{idx+1}")(skips[idx] * self.gamma)
            sample = up_block(sample + skip_in, latent_embeds)

        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample
    vae.decoder.forward = decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    return vae

# -------------------- paths & hyper‑params --------------------
root_day   = "/home/672511/contrastive-sdxl/data/ECP/train_A"
root_night = "/home/672511/contrastive-sdxl/data/ECP/train_B"
save_dir   = "/home/672511/contrastive-sdxl/models/vae/checkpoints"
os.makedirs(save_dir, exist_ok=True)

batch_size = 4
num_steps  = 20_000
lr         = 5e-5
warmup     = 500
device     = "cuda"
dtype      = torch.float32          

# -------------------- dataset --------------------
transform = transforms.Compose([
    transforms.Resize(int(512 * 1.25), transforms.InterpolationMode.BICUBIC),
    transforms.RandomCrop(512),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

ds_day   = FlatFolderDataset(root_day,   label=0, transform=transform)
ds_night = FlatFolderDataset(root_night, label=1, transform=transform)

vae_ds = ConcatDataset([ds_day, ds_night])

labels = torch.tensor([lbl for _, lbl in vae_ds])
class_counts = torch.bincount(labels)            # [len(day), len(night)]
weights = 1.0 / class_counts[labels]             # inverse frequency
sampler = WeightedRandomSampler(weights, num_samples=len(vae_ds), replacement=True)

dataloader = DataLoader(vae_ds, batch_size=batch_size,
                        sampler=sampler, num_workers=8,
                        pin_memory=True)

# -------------------- load & freeze --------------------
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-turbo",
                                    subfolder="vae",
                                    torch_dtype=dtype).to(device)
vae = patch_vae_for_skip(vae)

for p in vae.encoder.parameters():
    p.requires_grad_(False)
vae.decoder.conv_norm_out.requires_grad_(False)
vae.decoder.conv_out.requires_grad_(False)

# -------------------- optimisation --------------------
opt_params = [p for p in vae.parameters() if p.requires_grad]
optimiser  = optim.AdamW(opt_params, lr=lr, betas=(0.9, 0.99), weight_decay=0.01)
sched      = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=num_steps-warmup)

# -------------------- losses --------------------
mse_loss   = nn.MSELoss()
lpips_loss = LPIPS(net='vgg').to(device)
for p in lpips_loss.parameters():
    p.requires_grad_(False)

# -------------------- W&B --------------------
VIS_EVERY, NUM_SHOW = 50, 4
run = wandb.init(project="vae-decoder", name="decoder_finetune_fp32",
                 config=dict(batch_size=batch_size, lr=lr,
                             lpips_weight=0.5, vis_every=VIS_EVERY))

def denorm(x): return (x + 1) * 0.5

# -------------------- training loop --------------------
vae.train()
step = 0
pbar = tqdm(total=num_steps, desc="train-decoder")
while step < num_steps:
    for imgs, _ in dataloader:
        imgs = imgs.to(device, non_blocking=True)

        # ---------- forward ----------
        posterior = vae.encode(imgs).latent_dist
        latents   = posterior.sample()
        latents = latents * vae.config.scaling_factor

        vae.decoder.incoming_skip_acts = vae.encoder.current_down_blocks
        recon = vae.decode(latents/ vae.config.scaling_factor).sample
        recon.clamp(-1, 1)

        vae.decoder.incoming_skip_acts = None
        vae.encoder.current_down_blocks = None

        loss = mse_loss(recon, imgs) + 0.5 * lpips_loss(recon, imgs).mean()

        # ---------- backward ----------
        loss.backward()
        optimiser.step(); optimiser.zero_grad(set_to_none=True)
        if step > warmup: 
            sched.step()

        # ---------- W&B ----------
        if step % VIS_EVERY == 0:
            grid = make_grid(torch.cat([denorm(imgs[:NUM_SHOW]).cpu(),
                                        denorm(recon[:NUM_SHOW]).cpu()], 0),
                             nrow=NUM_SHOW)
            wandb.log({"recon_grid": wandb.Image(grid), "loss": loss.item()}, step=step)

        # ---------- checkpoint ----------
        if step % 500 == 0 and step:
            ckpt_path = os.path.join(save_dir, f"decoder_step{step:07d}.pt")
            torch.save(vae.decoder.state_dict(), ckpt_path)
            print(f"✅ saved {ckpt_path}")

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update(); step += 1
        if step >= num_steps: break
pbar.close()

torch.save(vae.decoder.state_dict(), os.path.join(save_dir, "decoder_final.pt"))
wandb.finish()
