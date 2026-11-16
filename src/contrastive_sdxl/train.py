import json
import os
from pathlib import Path

import torch
import lpips
import vision_aided_loss
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
import wandb

from .checkpoints import save_checkpoint
from .data.dataset import UnpairedDataset
from .data.labels import LabelBank, re_adjust_labels_to_random_crop512
from .losses.detector import YoloDetectionGuidance
from .losses.hdce import HDCELoss, calculate_HDCE_loss
from .losses.src import SRC_Loss, calculate_SRC_loss
from .model import Contrastive_SDXL, prepare_generator_paramters
from .models.dino import DINOAttentionHook, load_dino, resize_for_dino
from .models.patch_network import initialise_patch_projector
from .models.text import encode_prompt, load_text_encoders
from .models.unet import setup_unet
from .models.vae import VAE_encode, VAE_decode, setup_vae
from .schedules import make_sched


from .arguments import parse_args_unpaired_training


def optimizer_step(optimizer, scheduler, params, accelerator, max_grad_norm):
    accelerator.unscale_gradients(optimizer)
    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

def make_lr_scheduler(args, optimizer, name):
    return get_scheduler(name, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps,
                            num_training_steps=args.max_train_steps, num_cycles=args.lr_num_cycles,
                            power=args.lr_power)


def train(args):

    # if args.device == "cuda" and not torch.cuda.is_available():
        # raise RuntimeError("CUDA is unavailable. Run training on a GPU or select --device cpu.")
    # if args.train_batch_size != 1 or args.gradient_accumulation_steps != 1:
    #     raise ValueError("This training loop supports batch size 1 and accumulation steps 1.")

    output_dir = Path(args.output_dir).resolve()
    for name in ("checkpoints", "generated", "evaluations", "logs"):
        path = output_dir / name
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("YOLO_CONFIG_DIR", str(output_dir / "logs"/ "yolo_config"))

    accelerator = Accelerator(cpu=args.device == "cpu", mixed_precision=args.mixed_precision,
                              gradient_accumulation_steps=args.gradient_accumulation_steps,
                              log_with=None if args.report_to == "none" else args.report_to,
                              project_dir=str(output_dir / "logs"))
    
    if accelerator.num_processes != 1:
        raise ValueError("Use one process; this training script currently supports one process only.")
    
    set_seed(args.seed)
    device = accelerator.device

    if args.det_loss:
        if not Path(args.detector_weights).is_file():
            raise FileNotFoundError(f"Detector checkpoint not found: {args.detector_weights}")
        if args.train_img_prep != "sdxl_random_crop_512":
            raise ValueError("Detector guidance requires supports only'sdxl_random_crop_512'")
        
    dataset_train = UnpairedDataset(dataset_folder=args.dataset_folder, 
                                    image_prep=args.train_img_prep,
                                    split="train")
    train_dataloader = torch.utils.data.DataLoader(dataset_train, 
                                                   batch_size=args.train_batch_size, 
                                                   shuffle=True,
                                                   num_workers=args.dataloader_num_workers)
    
    logs_path = output_dir / "logs" / "training_args.json"
    logs_path.write_text(json.dumps(vars(args), indent=2) + "\n")


    tokenizers, text_encoders = load_text_encoders(device=device, revision=args.revision)

    unet, *unet_lora_target_modules = setup_unet(torch_dtype=torch.float32, rank=args.lora_rank_unet, 
                                                 return_lora_module_names=True)

    vae, vae_lora_target_modules = setup_vae(torch_dtype=torch.float32, rank=args.lora_rank_vae, 
                                             return_lora_module_names=True)

    discriminator = vision_aided_loss.Discriminator(cv_type="clip", loss_type=args.gan_loss_type, device=str(device))
    discriminator.cv_ensemble.requires_grad_(False)
    for name, module in discriminator.named_modules():
        if "attn" in name:
            module.fused_attn = False
            
    if is_xformers_available() and device.type == "cuda":
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("xFormers memory-efficient attention enabled")
        except (ValueError, RuntimeError) as error:
            print(f"Using standard attention: {error}")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    vae_enc, vae_dec = VAE_encode(vae), VAE_decode(vae)

    patch_projector = initialise_patch_projector(patch_projector=args.patch_projector, init_type=args.init_type, init_gain=args.init_gain, args=args)
    nce_layers = [int(i) for i in args.nce_layers.split(",")]
    
    fixed_caption_tgt = args.prompt
    fixed_prompt_embeds, fixed_pooled_embeds = encode_prompt(fixed_caption_tgt, tokenizers[0], tokenizers[1], 
                                                             text_encoders[0], text_encoders[1])
    del tokenizers, text_encoders
    if device.type == "cuda":
        torch.cuda.empty_cache()

    det_loss = label_bank = None
    if args.det_loss:
        det_loss = YoloDetectionGuidance(args.detector_weights, device)
        label_bank = LabelBank(args.labels_dir, device)
        missing = [Path(path).stem for path in dataset_train.l_imgs_src if Path(path).stem not in label_bank.bank]
        if missing:
            raise FileNotFoundError(f"Missing guidance labels for {len(missing)} images, including {missing[0]}")
        
    unet, vae_enc, vae_dec, discriminator, patch_projector = accelerator.prepare(unet, vae_enc, vae_dec, discriminator, patch_projector)

    noise_scheduler_1step = make_sched(device=device)

    dino = load_dino(device, args.dino_repo)
    attn_hook = DINOAttentionHook(dino, nce_layers)
    
    params_gen = prepare_generator_paramters(unet, vae)     
    optimizer_gen = torch.optim.AdamW(params_gen, 
                                      lr=args.learning_rate, 
                                      betas=(args.adam_beta1, args.adam_beta2), 
                                      weight_decay=args.adam_weight_decay, 
                                      eps=args.adam_epsilon)
                            
    params_disc = list(discriminator.parameters())
    optimizer_disc = torch.optim.AdamW(params_disc, 
                                       lr=args.learning_rate, 
                                       betas=(args.adam_beta1, args.adam_beta2), 
                                       weight_decay=args.adam_weight_decay, 
                                       eps=args.adam_epsilon)

    lr_scheduler_gen = make_lr_scheduler(args, optimizer_gen, args.lr_scheduler)
    lr_scheduler_disc = make_lr_scheduler(args, optimizer_disc, args.lr_scheduler)

    net_lpips = lpips.LPIPS(net="vgg").to(device).requires_grad_(False)

    net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(net_lpips, 
                                                                                                                            optimizer_gen, 
                                                                                                                            optimizer_disc, 
                                                                                                                            train_dataloader, 
                                                                                                                            lr_scheduler_gen, 
                                                                                                                            lr_scheduler_disc)
    
    accelerator.init_trackers(args.tracker_project_name, config=vars(args),
                              init_kwargs={"wandb": {"dir": str(output_dir / "logs")}})
    
    # Initialiase the Loses:
    src_loss = SRC_Loss(args)
    hDCE_loss = HDCELoss(args)
    crit_idt = torch.nn.L1Loss()

    optimizer_patch_projector = lr_scheduler_patch_projector = None

    use_idt_loss = args.dce_idt and args.lambda_idt > 0

    global_step = 0
    progress_bar = tqdm(total=args.max_train_steps, desc="Steps")
    metrics_path = output_dir / "logs" / "training.jsonl"

    try:
        for epoch in range(args.max_train_epochs):
            for batch in train_dataloader:

                img_a =  batch["pixel_values_src"]
                img_b = batch["pixel_values_tgt"]

                bsz = img_a.shape[0]

                time_ids_src  = batch["time_ids_src"]
                time_ids_tgt = batch["time_ids_tgt"]

                timesteps = torch.full((len(img_a),), 
                                       noise_scheduler_1step.config.num_train_timesteps - 1,
                                       device=device, 
                                       dtype=torch.long)
                
                dino_img_a = resize_for_dino(img_a)

                if global_step == 0:
                    with torch.no_grad(), accelerator.autocast():
                        # Keep the initialization forward and patch draw from the original loop
                        vae_enc(dino_img_a)
                        features = attn_hook.extract_attention_features(dino_img_a)
                        patch_projector(features, args.num_patches)
                    if args.patch_projector == "mlp_sample":
                        optimizer_patch_projector = torch.optim.AdamW(patch_projector.parameters(), 
                                                                      lr=args.learning_rate_patch_projector,
                                                                      betas=(args.adam_beta1_patch_projector, args.adam_beta2_patch_projector))
                                                           
                        lr_scheduler_patch_projector = make_lr_scheduler(args, 
                                                                         optimizer_patch_projector, 
                                                                         args.lr_scheduler_patch_projector)
                        
                        optimizer_patch_projector, lr_scheduler_patch_projector = accelerator.prepare(optimizer_patch_projector, 
                                                                                                      lr_scheduler_patch_projector)

                def translate(img, time_ids):
                    return Contrastive_SDXL.translate(
                        img, vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps,
                        fixed_prompt_embeds, fixed_pooled_embeds, time_ids)

                # ======================================================================================================
                # ---------------------------PW-CONTRASTIVE LOSS and Generator Objective (GAN)--------------------------
                # ======================================================================================================
                with accelerator.autocast():

                    fake_b = Contrastive_SDXL.translate(x = img_a, 
                                                        vae_enc = vae_enc, 
                                                        unet = unet, 
                                                        vae_dec = vae_dec, 
                                                        sched = noise_scheduler_1step, 
                                                        timesteps = timesteps, prompt_embeds = fixed_prompt_embeds, 
                                                        pooled_embeds = fixed_pooled_embeds, 
                                                        time_ids = time_ids_src)
                    
                    # idt_b = translate(img_b, time_ids_tgt) if use_idt_loss else None

                    dino_fake_b = resize_for_dino(fake_b)

                    real_a_features = attn_hook.extract_attention_features(img = dino_img_a)
                    fake_b_features = attn_hook.extract_attention_features(img = dino_fake_b)

                    fake_b_pool, sample_ids = patch_projector(feats=fake_b_features, num_patches=args.num_patches)
                    real_a_pool, _ = patch_projector(feats=real_a_features, num_patches=args.num_patches, patch_ids=sample_ids)

                    # Semantic Consistency Relation Loss
                    loss_SRC, weight = calculate_SRC_loss(src=real_a_pool, tgt=fake_b_pool, nce_layers=nce_layers,
                                                        src_loss=src_loss, args=args, step=global_step, batch_size=bsz ,epoch=epoch)

                    # HDCE Loss
                    if args.lambda_HDCE > 0.0:
                        loss_hDCE = calculate_HDCE_loss(src=real_a_pool, tgt=fake_b_pool, hdce_loss=hDCE_loss,args=args,
                                                            nce_layers=nce_layers, step = global_step, batch_size=bsz, weight=weight)
                    else:
                        loss_hDCE = fake_b.new_zeros(())

                # =====================================Object Consistency-Loss========================================    
                if args.det_loss and global_step >= 7000:
                    stems = batch["stem_src"]                              # list[str]
                    crop_info  = batch["time_ids_src"]                     # list[[H0,W0,i,j,512,512]]
                    flips = batch["flip_src"]                              # list[bool]


                    labels_list = []
                    for b, stem in enumerate(stems):
                        lab_orig_norm = label_bank.get(stem)               # (N,5) [cls,cx,cy,w,h] normalized to ORIGINAL
                        H0, W0, i, j, _, _ = crop_info[b].tolist()
                        lab_crop = re_adjust_labels_to_random_crop512(
                            labels_yolo_norm_orig=lab_orig_norm,
                            orig_hw=(int(H0), int(W0)),
                            crop_i=int(i), crop_j=int(j),
                            flip=bool(flips[b]),
                            resize_short=640,
                            stem=stem                              
                        )
                        
                        labels_list.append(lab_crop)

                    if any(lab is not None and lab.numel() > 0 for lab in labels_list):
                        with torch.autocast(device_type=fake_b.device.type, enabled=False):
                            loss_det = det_loss(fake_b.float(), labels_list) * args.lambda_det
                    else: # fallback: no boxes → set detector loss = 0.0
                        loss_det = torch.tensor(0.0, device=fake_b.device, requires_grad=False) 

                else:
                    loss_det = fake_b.new_zeros(())


                # ====================================================================================================
                # Backprop through contrastive losses and det loss
                accelerator.backward(loss_hDCE + loss_SRC + loss_det, retain_graph=False)

                # Remove FP16 scaling from the projector gradients before the generator
                # update can change the shared scale.
                if optimizer_patch_projector is not None:
                    accelerator.unscale_gradients(optimizer_patch_projector)

                optimizer_step(optimizer=optimizer_gen,
                               scheduler=lr_scheduler_gen,
                               params=params_gen,
                               accelerator=accelerator,
                               max_grad_norm=args.max_grad_norm)

                if optimizer_patch_projector is not None:
                    # Apply the current FP16 scale, which may have changed above.
                    if accelerator.scaler is not None:
                        for param in patch_projector.parameters():
                            if param.grad is not None:
                                param.grad.mul_(accelerator.scaler.get_scale())

                    # Update projector weights and learning rate and clear its gradients.
                    optimizer_patch_projector.step()
                    lr_scheduler_patch_projector.step()
                    optimizer_patch_projector.zero_grad(set_to_none=True)

                # =========================================Identity Loss===============================================
                loss_idt = fake_b.new_zeros(())

                if use_idt_loss:
                    with accelerator.autocast():
                        idt_b = Contrastive_SDXL.translate(x = img_b, 
                                                            vae_enc = vae_enc, 
                                                            unet = unet, 
                                                            vae_dec = vae_dec, 
                                                            sched = noise_scheduler_1step, 
                                                            timesteps = timesteps, prompt_embeds = fixed_prompt_embeds, 
                                                            pooled_embeds = fixed_pooled_embeds, 
                                                            time_ids = time_ids_tgt)
                        
                        loss_idt = crit_idt(img_b, idt_b) * args.lambda_idt
                        loss_idt = loss_idt + net_lpips(img_b, idt_b).mean() * args.lambda_idt_lpips


                    # Backprop through idt losses
                    accelerator.backward(loss_idt, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)

                    optimizer_gen.step()
                    lr_scheduler_gen.step()
                    optimizer_gen.zero_grad()

                # ======================================================================================================
                # ------------------------------------------- Generator Loss -------------------------------------------
                # ======================================================================================================
                with accelerator.autocast():
                    fake_b = Contrastive_SDXL.translate(x = img_a, 
                                    vae_enc = vae_enc, 
                                    unet = unet, 
                                    vae_dec = vae_dec, 
                                    sched = noise_scheduler_1step, 
                                    timesteps = timesteps, prompt_embeds = fixed_prompt_embeds, 
                                    pooled_embeds = fixed_pooled_embeds, 
                                    time_ids = time_ids_src)
                    # Compute adversarial loss (realisticness of fake_b)
                    loss_gan = discriminator(fake_b, for_G=True).mean() * args.lambda_gan

                accelerator.backward(loss_gan, retain_graph=False)

                # Backprop through GAN objective
                optimizer_step(optimizer_gen, lr_scheduler_gen, params_gen, accelerator, args.max_grad_norm)
                optimizer_disc.zero_grad(set_to_none=True)

                # ======================================================================================================
                # -------------------------Discriminator objective for task a->b(fake images) --------------------------
                # ======================================================================================================
                with accelerator.autocast():
                    loss_D_fake = discriminator(fake_b.detach(), for_real=False).mean() * args.lambda_gan

                accelerator.backward(loss_D_fake, retain_graph=False)
                optimizer_step(optimizer_disc, lr_scheduler_disc, params_disc, accelerator, args.max_grad_norm)

                # ======================================================================================================
                # ----------------------------Discriminator objective for real images ----------------------------------
                # ======================================================================================================
                with accelerator.autocast():
                    loss_D_real = discriminator(img_b, for_real=True).mean() * args.lambda_gan

                accelerator.backward(loss_D_real)
                optimizer_step(optimizer_disc, lr_scheduler_disc, params_disc, accelerator, args.max_grad_norm)


                logs = {"hdce":  loss_hDCE.detach().item(), 
                        "src": loss_SRC.detach().item(), 
                        "gan": loss_gan.detach().item(),
                        "disc": loss_D_fake.detach().item() + loss_D_real.detach().item(), 
                        "idt": loss_idt.detach().item(), 
                        "det": loss_det.detach().item()}
                
                global_step += 1
                progress_bar.update()
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                with metrics_path.open("a") as stream:
                    stream.write(json.dumps({"step": global_step, **logs}) + "\n")

                if (global_step - 1) % args.viz_freq == 0:
                    for tracker in accelerator.trackers:
                        if tracker.name == "wandb":
                            vis_log = {"train/real": [wandb.Image(img_a[idx].float().detach().cpu(), caption=f"idx={idx}") 
                                                       for idx in range(bsz)]}
                            vis_log["train/fake_b"] = [wandb.Image(fake_b[idx].float().detach().cpu(), caption=f"idx={idx}") 
                                                        for idx in range(bsz)]
                            tracker.log(vis_log, step=global_step)

                if (global_step - 1) % args.checkpointing_steps == 0:
                    checkpoint_filename = output_dir / "checkpoints" / "day2night" / f"model_{global_step}.pkl"
                    save_checkpoint(path=checkpoint_filename, 
                                    unet=accelerator.unwrap_model(unet), 
                                    vae_enc=accelerator.unwrap_model(vae_enc),
                                    vae_dec=accelerator.unwrap_model(vae_dec), 
                                    args=args, 
                                    unet_target_modules=unet_lora_target_modules, 
                                    vae_lora_target_modules=vae_lora_target_modules)
                    
                if global_step >= args.max_train_steps:
                    break
            if global_step >= args.max_train_steps:
                break
    finally:
        attn_hook.remove()
        progress_bar.close()
        accelerator.end_training()
    return global_step


def main(argv=None):
    train(parse_args_unpaired_training(argv))


if __name__ == "__main__":
    main()
