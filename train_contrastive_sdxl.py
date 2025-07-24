import os
import gc
import copy
import torch
import lpips
import wandb
from glob import glob
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers.optimization import get_scheduler
from peft.utils import get_peft_model_state_dict
from diffusers.utils.import_utils import is_xformers_available
# from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance
import vision_aided_loss
from utils.ddpm_utils import make_1step_sched,make_4step_sched
from constrastive_sdxl import ContrastiveSDXL_Turbo, VAE_encode, VAE_decode
from models.vae.vae_factory import initialize_vae
from models.unet.unet_factory import initialize_unet
from models.unet.unet_attention_hook import UNetAttentionHook
from models.dino_attn_hook import DINOAttentionHook, resize_for_dino
from models.patch_network import define_F
from losses.zeroFlow import IdentityWarpLoss, lambda_warp
from losses.SRC import SRC_Loss, calculate_R_loss
from losses.hDCE import PatchHDCELoss, calculate_HDCE_loss
from utils.train_utils import UnpairedDataset, build_transform, parse_args_unpaired_training, data_dependent_initialize, prep_for_roma, contrastive_at_scale
from utils import setup_paths
from romatch import roma_outdoor
from diffusers import DDPMScheduler


# from my_utils.dino_struct import DinoStructureLoss


def train(args):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to
        # mixed_precision="bf16"
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # === SDXL-Turbo uses two tokenizers ===
    tokenizer_1 = AutoTokenizer.from_pretrained("stabilityai/sdxl-turbo", subfolder="tokenizer", revision=args.revision,
                                                use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained("stabilityai/sdxl-turbo", subfolder="tokenizer_2",
                                                revision=args.revision, use_fast=False)

    # === SDXL-Turbo uses two text encoders ===
    text_encoder_1 = CLIPTextModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("stabilityai/sdxl-turbo",
                                                                 subfolder="text_encoder_2")

    # === DDPM scheduler & Timestep for 1-step DDPM === (denoise at timestep t, e.g. 239)

    scheduler = make_1step_sched()
    timestep = 239

    # === Initialize UNet and VAE with LoRA setup ===
    unet, l_modules_unet_encoder, l_modules_unet_decoder, l_modules_unet_others = initialize_unet(
        args.lora_rank_unet, return_lora_module_names=True
    )

    vae, vae_lora_target_modules = initialize_vae(args.lora_rank_vae, return_lora_module_names=True)

    # === We do not train the text encoders ===
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    if args.gan_disc_type == "vagan_clip":
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc.cv_ensemble.requires_grad_(False)

    # Attempt to turn on xFormers only if installed
    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("🚀 xFormers memory-efficient attention enabled")
        except Exception as e:
            print(f"⚠️ Could not enable xFormers attention: {e}")
    else:
        print("⚙️ xFormers not installed; using standard attention")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    unet.conv_in.requires_grad_(True)

    vae_enc = VAE_encode(vae)
    vae_dec = VAE_decode(vae)

    # Create training dataset and dataloader with SDXL-compatible tokenizer
    dataset_train = UnpairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.train_img_prep,
        split="train"
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers
    )

    # Validation transforms and static captions
    T_val = build_transform(args.val_img_prep)
    fixed_caption_src = dataset_train.fixed_caption_src
    fixed_caption_tgt = dataset_train.fixed_caption_tgt

    # Gather all validation images from both domains
    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_src_test.extend(glob(os.path.join(args.dataset_folder, "test_A", ext)))
    l_images_tgt_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_tgt_test.extend(glob(os.path.join(args.dataset_folder, "test_B", ext)))
    l_images_src_test, l_images_tgt_test = sorted(l_images_src_test), sorted(l_images_tgt_test)

    # TODO: Add validation logic using l_images_src_test and l_images_tgt_test

    netF = define_F(netF=args.netF, init_type=args.init_type, init_gain=args.init_gain, args=args)
    nce_layers = [int(i) for i in args.nce_layers.split(',')]

    text_encoder_1, text_encoder_2 = accelerator.prepare(text_encoder_1, text_encoder_2)

    # Tokenize and embed fixed caption for A→B direction
    fixed_prompt_tokens_1 = tokenizer_1(
        fixed_caption_tgt,
        max_length=tokenizer_1.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(accelerator.device)

    fixed_prompt_tokens_2 = tokenizer_2(
        fixed_caption_tgt,
        max_length=tokenizer_2.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(accelerator.device)

    with torch.no_grad():
        out_1 = text_encoder_1(fixed_prompt_tokens_1, output_hidden_states=True)
        out_2 = text_encoder_2(fixed_prompt_tokens_2, output_hidden_states=True)

        fixed_prompt_embeds = torch.cat([out_1.hidden_states[-2], out_2.hidden_states[-2]], dim=-1)
        fixed_pooled_embeds = out_2[0]

    # free up memory
    del text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2
    torch.cuda.empty_cache()

    # Initialize RoMa model:
    roma = roma_outdoor(device='cuda').eval()
    for p in roma.parameters():
        p.requires_grad = False
    roma.decoder.detach = False    
    vit = roma.encoder.dinov2_vitl14[0]
    vit.forward_features = torch.enable_grad()(vit.forward_features)
    roma.match = roma.match.__wrapped__.__get__(roma, type(roma))  # Grab the original method to avoid issues with `torch.no_grad()`

    # Prepare models, optimizers, dataloader, and schedulers with Accelerator
    unet, vae_enc, vae_dec, net_disc, netF, roma = accelerator.prepare(unet, vae_enc, vae_dec, net_disc, netF, roma)
    
    # Prepare trainable parameters and VAE wrappers
    params_gen = ContrastiveSDXL_Turbo.get_trainable_params(unet, vae)

    # Initialiase CUT the Loses:
    src_loss = SRC_Loss(args)
    hDCE_loss = PatchHDCELoss(args)

    # Initialiase the Loses:
    iwarp_loss = IdentityWarpLoss()
    crit_idt = torch.nn.L1Loss()

    # === Attach attention feature hook ===
    # --- DINO Attention Hook ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    dino = dino.to(device)
    attn_hook = DINOAttentionHook(dino, nce_layers=nce_layers)
    
    # Optimizer for generator components (UNet + LoRA + VAE LoRA + skip convs)
    optimizer_gen = torch.optim.AdamW(
        params_gen,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Optimizer for the discriminator
    params_disc = list(net_disc.parameters())
    optimizer_disc = torch.optim.AdamW(
        params_disc,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Learning rate schedulers for generator 
    lr_scheduler_gen = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_gen,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power
    )
    
    # Learning rate schedulers for discriminator
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power
    )

    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.cuda()
    for p in net_lpips.parameters():
        p.requires_grad_(False)

    
    net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
        net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    first_epoch = 0
    global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Turn off efficient attention in the discriminator (if any)
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    for epoch in range(first_epoch, args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # Accumulate gradients over these modules
            l_acc = [unet, net_disc, vae_enc, vae_dec]
            with accelerator.accumulate(*l_acc):
                # Load images and cast to correct dtype
                img_a = batch["pixel_values_src"]
                img_b = batch["pixel_values_tgt"]

                bsz = img_a.shape[0]

                time_ids_src = torch.tensor(batch["time_ids_src"], device=img_a.device).unsqueeze(0).repeat(bsz, 1)
                time_ids_tgt = torch.tensor(batch["time_ids_tgt"], device=img_a.device).unsqueeze(0).repeat(bsz, 1)

                fixed_prompt_embeds = fixed_prompt_embeds.repeat(bsz, 1, 1)
                fixed_pooled_embeds = fixed_pooled_embeds.repeat(bsz, 1)
                
                dino_img_a = resize_for_dino(img_a)
                # First-time initialization of netF (MLP heads) and its optimizer
                if epoch == 0 and global_step == 0:
                    optimizer_netF = data_dependent_initialize(data=dino_img_a, netF=netF, unet=unet, vae_enc=vae_enc,
                                                               attn_hook=attn_hook, nce_layers=nce_layers, args=args,
                                                               accelerator=accelerator
                                                               )
                    lr_scheduler_netF = get_scheduler(
                        args.lr_scheduler_netF,
                        optimizer=optimizer_netF,
                        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                        num_training_steps=args.max_train_steps * accelerator.num_processes,
                        num_cycles=args.lr_num_cycles,
                        power=args.lr_power
                    )

                    optimizer_netF, lr_scheduler_netF = accelerator.prepare(optimizer_netF, lr_scheduler_netF)


                # ======================================================================================================
                # ---------------------------PW-CONTRASTIVE LOSS and Generator Objective (GAN)--------------------------
                # ======================================================================================================

                # encode img_a
                lat_a = ContrastiveSDXL_Turbo.encode(img_a, vae_enc)

                # Denoise using UNet of SDXL-Turbo.
                fake_lat_a, _ = ContrastiveSDXL_Turbo.denoise(latents=lat_a, unet=unet,
                                                                                        scheduler=scheduler,
                                                                                        timestep = timestep,
                                                                                        prompt_embeds=fixed_prompt_embeds,
                                                                                        pooled_embeds=fixed_pooled_embeds,
                                                                                        time_ids=time_ids_src)

                # decode back to pixels
                fake_b = ContrastiveSDXL_Turbo.decode(fake_lat_a, vae_dec)

                # encode img_b
                lat_b = ContrastiveSDXL_Turbo.encode(img_b, vae_enc)

                # Denoise 
                idt_lat_b, _ = ContrastiveSDXL_Turbo.denoise(latents=lat_b, unet=unet,
                                                                        scheduler=scheduler,
                                                                        timestep = timestep,
                                                                        prompt_embeds=fixed_prompt_embeds,
                                                                        pooled_embeds=fixed_pooled_embeds,
                                                                        time_ids=time_ids_tgt)

                # decode back to pixels
                idt_b = ContrastiveSDXL_Turbo.decode(idt_lat_b, vae_dec)

                dino_fake_b = resize_for_dino(fake_b)
                real_a_features = attn_hook.extract_attention_features(img = dino_img_a)   
                fake_b_features = attn_hook.extract_attention_features(img = dino_fake_b)

                # Sample contrastive patches using netF (MLPs)
                fake_b_pool, sample_ids = netF(feats=fake_b_features, num_patches=args.num_patches, patch_ids=None)
                real_a_pool, _ = netF(feats=real_a_features, num_patches=args.num_patches, patch_ids=sample_ids)

                # Semantic Consistency Relation Loss
                loss_SRC, weight = calculate_R_loss(src=real_a_pool, tgt=fake_b_pool, nce_layers=nce_layers,
                                                    src_loss=src_loss, args=args, epoch=epoch)
                
                # HDCE Loss
                if args.lambda_HDCE > 0.0:
                    loss_hDCE = calculate_HDCE_loss(src=real_a_pool, tgt=fake_b_pool, hDCE=hDCE_loss,args=args,
                                                    nce_layers=nce_layers, weight=weight)
                else:
                    loss_hDCE = 0.0

                # =====================================ZeroFLow-Loss========================================
                source = prep_for_roma(img_a) # Prepare source image for RoMa
                target = prep_for_roma(fake_b) # Prepare target image for RoMa

                warp, cert = roma.match(source, target, device=accelerator.device)

                loss_iwarp = iwarp_loss(warp, cert)

                lambda_w = lambda_warp(global_step, args)

                zeroflow_loss = lambda_w * loss_iwarp
                
                # =====================================ZeroFLow-Loss========================================

                # =======================================Identity-Loss======================================
                loss_idt = 0
                if args.dce_idt and args.lambda_idt > 0.0:
                    loss_idt = crit_idt(img_b, idt_b) * args.lambda_idt
                    loss_idt += net_lpips(img_b, idt_b).mean() * args.lambda_idt_lpips

                # Backprop through all loses
                accelerator.backward(loss_hDCE + loss_SRC + zeroflow_loss + loss_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)

                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                if accelerator.sync_gradients:
                    optimizer_netF.step()
                    lr_scheduler_netF.step()
                optimizer_netF.zero_grad()

                # ======================================================================================================
                # ------------------------------------------- Generator Loss -------------------------------------------

                lat_a = ContrastiveSDXL_Turbo.encode(img_a, vae_enc)

                fake_lat_a, _ = ContrastiveSDXL_Turbo.denoise(latents=lat_a, unet=unet,
                                                              scheduler=scheduler,
                                                              timestep = timestep,
                                                              prompt_embeds=fixed_prompt_embeds,
                                                              pooled_embeds=fixed_pooled_embeds,
                                                              time_ids=time_ids_src)

                fake_b = ContrastiveSDXL_Turbo.decode(fake_lat_a, vae_dec)


                # Compute adversarial loss (realisticness of fake_b)
                loss_gan = net_disc(fake_b, for_G=True).mean() * args.lambda_gan
                accelerator.backward(loss_gan, retain_graph=False)

                # Backprop through GAN objective
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)

                # Apply updates
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()

                if accelerator.sync_gradients:
                    optimizer_netF.step()
                    lr_scheduler_netF.step()
                optimizer_netF.zero_grad()

                # ======================================================================================================
                # -------------------------Discriminator objective for task a->b(fake images) --------------------------
                # ======================================================================================================
                loss_D_fake = net_disc(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(loss_D_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                # ======================================================================================================
                # ----------------------------Discriminator objective for real images ----------------------------------
                # ======================================================================================================
                loss_D_real = net_disc(img_b, for_real=True).mean() * args.lambda_gan
                accelerator.backward(loss_D_real, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                logs = {}
                logs["hDCE"] = loss_hDCE.detach().item()
                logs["SRC"] = loss_SRC.detach().item()
                logs["iwarp"] = zeroflow_loss.detach().item()
                logs["gan"] = loss_gan.detach().item()
                logs["disc"] = loss_D_fake.detach().item() + loss_D_real.detach().item()
                logs["idt"] = loss_idt.detach().item()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        eval_unet = accelerator.unwrap_model(unet)
                        eval_vae_enc = accelerator.unwrap_model(vae_enc)
                        eval_vae_dec = accelerator.unwrap_model(vae_dec)
                        if global_step % args.viz_freq == 1:
                            for tracker in accelerator.trackers:
                                if tracker.name == "wandb":
                                    viz_img_a = batch["pixel_values_src"]
                                    viz_img_b = batch["pixel_values_tgt"]
                                    log_dict = {
                                        "train/real": [
                                            wandb.Image(viz_img_a[idx].float().detach().cpu(), caption=f"idx={idx}") for
                                            idx in range(bsz)],
                                        # "train/real_b": [
                                        #     wandb.Image(viz_img_b[idx].float().detach().cpu(), caption=f"idx={idx}") for
                                        #     idx in range(bsz)],
                                    }
                                    log_dict["train/fake_b"] = [
                                        wandb.Image(fake_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in
                                        range(bsz)]
                                    tracker.log(log_dict)
                                    gc.collect()

                        if global_step % args.checkpointing_steps == 1:
                            outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                            sd = {}
                            sd["l_target_modules_encoder"] = l_modules_unet_encoder
                            sd["l_target_modules_decoder"] = l_modules_unet_decoder
                            sd["l_modules_others"] = l_modules_unet_others
                            sd["rank_unet"] = args.lora_rank_unet
                            sd["sd_encoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_encoder")
                            sd["sd_decoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_decoder")
                            sd["sd_other"] = get_peft_model_state_dict(eval_unet, adapter_name="default_others")
                            sd["rank_vae"] = args.lora_rank_vae
                            sd["vae_lora_target_modules"] = vae_lora_target_modules
                            sd["sd_vae_enc"] = eval_vae_enc.state_dict()
                            sd["sd_vae_dec"] = eval_vae_dec.state_dict()
                            torch.save(sd, outf)
                            gc.collect()
                            torch.cuda.empty_cache()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break


if __name__ == "__main__":
    args = parse_args_unpaired_training()
    train(args)
