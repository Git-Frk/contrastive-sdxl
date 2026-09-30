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
from pathlib import Path
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers.optimization import get_scheduler
from peft.utils import get_peft_model_state_dict
from diffusers.utils.import_utils import is_xformers_available
# from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance
import vision_aided_loss
from utils.ddpm_utils import make_sched
from constrastive_sdxl import ContrastiveSDXL_Turbo, VAE_encode, VAE_decode
from models.vae.vae_factory import initialize_vae
from models.unet.unet_factory import initialize_unet
from models.unet.unet_attention_hook import UNetAttentionHook
from models.dino_attn_hook import DINOAttentionHook, resize_for_dino
from models.patch_network import define_F
from losses.SRC import SRC_Loss, calculate_R_loss
from losses.hDCE import PatchHDCELoss, calculate_HDCE_loss
from losses.detector_loss import YoloDetectionGuidance, LabelBank, re_adjust_labels_to_random_crop512
from utils.train_utils import UnpairedDataset, build_transform, parse_args_unpaired_training, data_dependent_initialize, prep_for_roma, ScheduleLambda, LinearSchedule
from utils import setup_paths
# from romatch import roma_outdoor
from ultralytics import YOLO


# from my_utils.dino_struct import DinoStructureLoss


def train(args):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to
    )
    set_seed(args.seed)
    grad_accum = args.gradient_accumulation_steps

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # === SDXL-Turbo uses two tokenizers ===
    tokenizer_1 = AutoTokenizer.from_pretrained("stabilityai/sdxl-turbo", subfolder="tokenizer", revision=args.revision,
                                                use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained("stabilityai/sdxl-turbo", subfolder="tokenizer_2",
                                                revision=args.revision, use_fast=False)

    # === SDXL-Turbo uses two text encoders ===
    text_encoder_1 = CLIPTextModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("stabilityai/sdxl-turbo",
                                                                 subfolder="text_encoder_2")

    # === 1-step DDIM scheduler ===
    noise_scheduler_1step = make_sched()

    # === Initialize UNet and VAE with LoRA setup ===
    unet, l_modules_unet_encoder, l_modules_unet_decoder, l_modules_unet_others = initialize_unet(weight_dtype,
        args.lora_rank_unet, return_lora_module_names=True
    )

    vae, vae_lora_target_modules = initialize_vae(weight_dtype, args.lora_rank_vae, return_lora_module_names=True)

    # === The text encoders are not trained ===
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    if args.gan_disc_type == "vagan_clip":
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc.cv_ensemble.requires_grad_(False)

    # Initialiase the Loses:
    src_loss = SRC_Loss(args)
    hDCE_loss = PatchHDCELoss(args)
    crit_idt = torch.nn.L1Loss()

    # Attempt to turn on xFormers only if installed
    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("🚀 xFormers memory-efficient attention enabled")
        except Exception as e:
            print(f"⚠️  Could not enable xFormers attention: {e}")
    else:
        print("! xFormers not installed; using standard attention")

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

    fixed_caption_tgt = dataset_train.fixed_caption_tgt

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

    del text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2
    torch.cuda.empty_cache()

    # Initialize YOLO pretrained model and the detector_loss
    weights = "/home/672511/contrastive-sdxl/models/yolo11_det/runs/detect/train_ecp_yolo11/weights/best.pt"
    yolo_model = YOLO(weights)

    # Intialising Label banks
    labels_dir = Path("/home/672511/contrastive-sdxl/models/yolo11_det/datasets/loss-labels/labels/annotations")  
    label_bank = LabelBank(labels_dir, device="cuda:0")

    # Prepare models, optimizers, dataloader, and schedulers with Accelerator
    unet, vae_enc, vae_dec, net_disc, netF, yolo_model = accelerator.prepare(unet, vae_enc, vae_dec, net_disc, netF, yolo_model)

    #Initialising the detector guidance loss:
    det_loss = YoloDetectionGuidance(yolo_model)

    # === Attach attention feature hook ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    dino = dino.to(device)
    attn_hook = DINOAttentionHook(dino, nce_layers=nce_layers)

    # Prepare trainable parameters and VAE wrappers
    params_gen = ContrastiveSDXL_Turbo.get_trainable_params(unet, vae)
    
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
    net_lpips.requires_grad_(False)
    
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

                # Timesteps for 1-step DDIM
                timesteps = torch.tensor(
                    [noise_scheduler_1step.config.num_train_timesteps - 1] * bsz,
                    device=img_a.device
                ).long()
                
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

                fake_lat_a = ContrastiveSDXL_Turbo.denoise(latents=lat_a, unet=unet,
                                                           scheduler=noise_scheduler_1step,
                                                           timestep=timesteps,
                                                           prompt_embeds=fixed_prompt_embeds,
                                                           pooled_embeds=fixed_pooled_embeds,
                                                           time_ids=time_ids_src
                                                        )

                # decode back to pixels
                fake_b = ContrastiveSDXL_Turbo.decode(fake_lat_a, vae_dec)
                
                use_idt_loss = args.dce_idt and args.lambda_idt > 0.0 # turn idt loss on or off 
                if use_idt_loss:
                    # encode img_b
                    lat_b = ContrastiveSDXL_Turbo.encode(img_b, vae_enc)

                    idt_lat_b = ContrastiveSDXL_Turbo.denoise(latents=lat_b, unet=unet,
                                                              scheduler=noise_scheduler_1step,
                                                              timestep=timesteps,
                                                              prompt_embeds=fixed_prompt_embeds,
                                                              pooled_embeds=fixed_pooled_embeds,
                                                              time_ids=time_ids_tgt
                                                             )

                    # decode back to pixels
                    idt_b = ContrastiveSDXL_Turbo.decode(idt_lat_b, vae_dec)

                # Extract self-attention feature maps from fake_B (translated) image - using DINOv2.
                dino_fake_b = resize_for_dino(fake_b)
                real_a_features = attn_hook.extract_attention_features(img = dino_img_a)   
                fake_b_features = attn_hook.extract_attention_features(img = dino_fake_b)

                # Sample contrastive patches using netF (MLPs)
                fake_b_pool, sample_ids = netF(feats=fake_b_features, num_patches=args.num_patches, patch_ids=None)
                real_a_pool, _ = netF(feats=real_a_features, num_patches=args.num_patches, patch_ids=sample_ids)

                # Semantic Consistency Relation Loss
                loss_SRC, weight = calculate_R_loss(src=real_a_pool, tgt=fake_b_pool, nce_layers=nce_layers,
                                                        src_loss=src_loss, args=args, step = global_step, epoch=epoch)
                
                # HDCE Loss
                if args.lambda_HDCE > 0.0:
                    loss_hDCE = calculate_HDCE_loss(src=real_a_pool, tgt=fake_b_pool, hDCE=hDCE_loss,args=args,
                                                        nce_layers=nce_layers, step = global_step, weight=weight)
                else:
                    loss_hDCE = 0.0
                
                # =====================================Object Consistency-Loss========================================
                if args.det_loss and global_step>6999:

                    stems = batch["stem_src"]                              # list[str]
                    crop_info  = batch["time_ids_src"]                     # list[[H0,W0,i,j,512,512]]
                    flips = batch["flip_src"]                              # list[bool]

                    labels_list = []
                    for b, stem in enumerate(stems):
                        lab_orig_norm = label_bank.get(stem)               # (N,5) [cls,cx,cy,w,h] normalized to ORIGINAL
                        H0, W0, i, j, _, _ = crop_info
                        lab_crop = re_adjust_labels_to_random_crop512(
                            labels_yolo_norm_orig=lab_orig_norm,
                            orig_hw=(int(H0), int(W0)),
                            crop_i=int(i), crop_j=int(j),
                            flip=bool(flips[b]),
                            resize_short=640,
                            stem=stem                              
                        )
                        # labels_list.append(lab_crop)
                        if lab_crop is not None:       # skip empty crops
                            labels_list.append(lab_crop)

                    # detector input should be in [0,1]
                    if labels_list: # only compute if non-empty
                        det_in = (fake_b.clamp(-1, 1) + 1.0) * 0.5
                        loss_det = det_loss(det_in, labels_list) * args.lambda_det
                    else: # fallback: no boxes → set detector loss = 0.0
                        loss_det = torch.tensor(0.0, device=fake_b.device, requires_grad=False) 
                    
                else:
                    loss_det = torch.tensor(0.0, device=fake_b.device)
                    
                # Backprop through contrastive losses and det loss
                # accelerator.backward(loss_hDCE + loss_SRC + loss_det, retain_graph=False)
                accelerator.backward((loss_hDCE + loss_SRC + loss_det)/grad_accum, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)

                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                if accelerator.sync_gradients:
                    optimizer_netF.step()
                    lr_scheduler_netF.step()
                optimizer_netF.zero_grad()              
                # =====================================Object Consistency-Loss========================================

                # =========================================Identity Loss===============================================
                loss_idt = torch.tensor(0.0, device=img_b.device)
                if use_idt_loss:
                    loss_idt = crit_idt(img_b, idt_b) * args.lambda_idt
                    loss_idt += net_lpips(img_b, idt_b).mean() * args.lambda_idt_lpips

                    # Backprop through idt losses
                    accelerator.backward(loss_idt, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)

                    optimizer_gen.step()
                    lr_scheduler_gen.step()
                    optimizer_gen.zero_grad()
                # =========================================Identity Loss===============================================


                # ======================================================================================================
                # ------------------------------------------- Generator Loss -------------------------------------------

                lat_a = ContrastiveSDXL_Turbo.encode(img_a, vae_enc)

                fake_lat_a = ContrastiveSDXL_Turbo.denoise(latents=lat_a, unet=unet,
                                                           scheduler=noise_scheduler_1step,
                                                           timestep=timesteps,
                                                           prompt_embeds=fixed_prompt_embeds,
                                                           pooled_embeds=fixed_pooled_embeds,
                                                           time_ids=time_ids_src
                                                          )

                fake_b = ContrastiveSDXL_Turbo.decode(fake_lat_a, vae_dec)

                # Compute adversarial loss (realisticness of fake_b)
                loss_gan = net_disc(fake_b, for_G=True).mean() * args.lambda_gan

                accelerator.backward(loss_gan/grad_accum, retain_graph=False)

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
                accelerator.backward(loss_D_fake/grad_accum, retain_graph=False)
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
                accelerator.backward(loss_D_real/grad_accum, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                logs = {}
                logs["hDCE"] = loss_hDCE.detach().item()
                logs["SRC"] = loss_SRC.detach().item()
                logs["gan"] = loss_gan.detach().item()
                logs["disc"] = loss_D_fake.detach().item() + loss_D_real.detach().item()
                logs["idt"] = loss_idt.detach().item()
                logs["det"] = loss_det.detach().item()
                
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

