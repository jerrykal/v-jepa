# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel

from app.diffusion_decoder.transforms import denormalize_clips, make_transforms
from app.diffusion_decoder.utils import (
    get_pretrained_vae,
    init_models,
    init_opt,
    load_checkpoint,
    load_jepa_encoder,
)
from src.datasets.data_manager import init_data
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import (
    AverageMeter,
    CSVLogger,
    adamw_logger,
    get_logger,
    gpu_timer,
    grad_logger,
)

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    gradient_checkpointing = cfgs_meta.get("gradient_checkpointing", False)
    which_dtype = cfgs_meta.get("dtype")
    pre_train_model = cfgs_meta.get("pre_train_model", None)
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MODEL
    cfgs_model = args.get("model")
    model_name = cfgs_model.get("model_name")
    uniform_power = cfgs_model.get("uniform_power", True)

    # -- DIFFUSION
    cfgs_diffusion = args.get("diffusion")
    in_channels = cfgs_diffusion.get("in_channels", 4)
    out_channels = cfgs_diffusion.get("out_channels", 4)
    sample_size = cfgs_diffusion.get("sample_size", 64)
    layers_per_block = cfgs_diffusion.get("layers_per_block", 2)
    block_out_channels = cfgs_diffusion.get(
        "block_out_channels", (320, 640, 1280, 1280)
    )
    down_block_types = cfgs_diffusion.get(
        "down_block_types",
        (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
    )
    up_block_types = cfgs_diffusion.get(
        "up_block_types",
        (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
    )
    scheduler_beta_start = cfgs_diffusion.get("scheduler_beta_start", 0.00085)
    scheduler_beta_end = cfgs_diffusion.get("scheduler_beta_end", 0.012)
    scheduler_beta_schedule = cfgs_diffusion.get(
        "scheduler_beta_schedule", "scaled_linear"
    )
    scheduler_prediction_type = cfgs_diffusion.get(
        "scheduler_prediction_type", "epsilon"
    )
    do_latent_diffusion = cfgs_diffusion.get("do_latent_diffusion", True)
    vae_model_id = cfgs_diffusion.get("vae_model_id", None)
    assert not do_latent_diffusion or vae_model_id is not None, (
        "VAE model must be provided if latent diffusion is enabled"
    )

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights", None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), (
            "Must have one sampling weight specified for each dataset"
        )
    batch_size = cfgs_data.get("batch_size")
    num_clips = cfgs_data.get("num_clips")
    num_frames = cfgs_data.get("num_frames")
    tubelet_size = cfgs_data.get("tubelet_size")
    sampling_rate = cfgs_data.get("sampling_rate")
    duration = cfgs_data.get("clip_duration", None)
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    filter_short_videos = cfgs_data.get("filter_short_videos", False)
    decode_one_clip = cfgs_data.get("decode_one_clip", True)
    log_resource_util_data = cfgs_data.get("log_resource_utilization", False)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    lr = cfgs_opt.get("lr")
    lr_scheduler_type = cfgs_opt.get("lr_scheduler_type", "constant")
    clip_grad = cfgs_opt.get("clip_grad", None)
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    wd = float(cfgs_opt.get("weight_decay"))
    eps = cfgs_opt.get("eps", 1.0e-8)

    # -- LOGGING
    cfgs_logging = args.get("logging")
    folder = cfgs_logging.get("folder")
    tag = cfgs_logging.get("write_tag")

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    latest_file = f"{tag}-latest.pth.tar"
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "grad-norm"),
        ("%d", "gpu-time(ms)"),
        ("%d", "wall-time(ms)"),
    )

    # -- init model
    encoder, unet, noise_scheduler = init_models(
        device=device,
        uniform_power=uniform_power,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        use_sdpa=use_sdpa,
        in_channels=in_channels,
        out_channels=out_channels,
        sample_size=sample_size,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        scheduler_beta_start=scheduler_beta_start,
        scheduler_beta_end=scheduler_beta_end,
        scheduler_beta_schedule=scheduler_beta_schedule,
        scheduler_prediction_type=scheduler_prediction_type,
    )
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # -- make data transforms
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        clip_len=num_frames,
        frame_sample_rate=sampling_rate,
        filter_short_videos=filter_short_videos,
        decode_one_clip=decode_one_clip,
        duration=duration,
        num_clips=num_clips,
        transform=transform,
        datasets_weights=datasets_weights,
        collator=None,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        rank=rank,
        log_dir=folder if log_resource_util_data else None,
    )
    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataest length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler
    optimizer, scaler, lr_scheduler = init_opt(
        models=[encoder, unet],
        scheduler_type=lr_scheduler_type,
        lr=lr,
        wd=wd,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )

    # -- freeze encoder
    for p in encoder.parameters():
        p.requires_grad_(False)

    start_epoch = 0

    # -- load pre-trained encoder
    assert pre_train_model is not None, "Pre-trained encoder must be provided"
    encoder = load_jepa_encoder(pre_train_model, encoder)

    # -- load pre-trained vae
    if do_latent_diffusion:
        vae = get_pretrained_vae(vae_model_id, device)
        vae.requires_grad_(False)

    # -- load training checkpoint
    if load_model:
        (
            unet,
            noise_scheduler,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            unet=unet,
            noise_scheduler=noise_scheduler,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            lr_scheduler.step()

    # -- wrap unet with DDP
    unet = DistributedDataParallel(unet, static_graph=True)

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "unet": unet.state_dict(),
            "unet_config": unet.module.config,
            "noise_scheduler_config": noise_scheduler.config,
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        unsupervised_sampler.set_epoch(start_epoch)
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                udata = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                udata = next(loader)

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        input_var_meter = AverageMeter()
        input_var_min_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        wall_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            try:
                udata = next(loader)
            except Exception:
                logger.info("Exhausted data loaders. Refreshing...")
                loader = iter(unsupervised_loader)
                udata = next(loader)

            # -- unsupervised video clips
            # Put each clip on the GPU and concatenate along batch
            # dimension, resulting in (B, C, F, H, W)
            clips = torch.cat(
                [u.to(device, non_blocking=True) for u in udata[0]], dim=0
            )

            def train_step():
                lr_scheduler.step()
                _new_lr = lr_scheduler.get_last_lr()[0]

                # Step 1. Forward
                with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                    encoder_hidden_states = encoder(clips)
                    encoder_hidden_states = F.layer_norm(
                        encoder_hidden_states, (encoder_hidden_states.size(-1),)
                    )

                    # (B, L, D) -> (B * num_frames, L, D), L is the number of patches, D is the embedding dimension
                    encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                        num_frames, dim=0
                    )

                    # Denormalize clips to range [0, 1] & reshape to a larger batch of images
                    images = denormalize_clips(clips.clone())
                    images = rearrange(images, "b c f h w -> (b f) c h w")

                    # Normalize clips to range (-1, 1)
                    images = images * 2.0 - 1.0

                    # Convert images to latent space if a VAE encoder is provided
                    if do_latent_diffusion:
                        latents = vae.encode(images).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
                    else:
                        latents = images

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents, device=device)

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=device,
                    ).long()

                    # Create class labels indicating which frame we are reconstructing
                    class_labels = torch.arange(num_frames, device=device)
                    class_labels = class_labels.repeat(batch_size)

                    # Add noise to the clean latents
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the model prediction
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states,
                        class_labels,
                        return_dict=False,
                    )[0]

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unsupported prediction type: {noise_scheduler.config.prediction_type}"
                        )

                    # Compute the loss
                    loss = F.mse_loss(
                        noise_pred.float(), target.float(), reduction="mean"
                    )

                # Step 2. Backward & step
                _dec_norm = 0.0
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if clip_grad is not None:
                    _dec_norm = torch.nn.utils.clip_grad_norm_(
                        unet.parameters(), clip_grad
                    )
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                grad_stats = grad_logger(unet.named_parameters())
                grad_stats.global_norm = float(_dec_norm)
                optimizer.zero_grad()
                optim_stats = adamw_logger(optimizer)

                return (
                    float(loss),
                    _new_lr,
                    grad_stats,
                    optim_stats,
                )

            (
                (
                    loss,
                    _new_lr,
                    grad_stats,
                    optim_stats,
                ),
                gpu_etime_ms,
            ) = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            input_var = float(
                AllReduce.apply(clips.view(clips.shape[0], -1).var(dim=1).mean(dim=0))
            )
            input_var_min = float(
                AllReduce.apply(torch.min(clips.view(clips.shape[0], -1).var(dim=1)))
            )
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    grad_stats.global_norm,
                    gpu_etime_ms,
                    iter_elapsed_time_ms,
                )
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f | "
                        "input_var: %.3f %.3f | "
                        "[lr: %.2e] "
                        "[mem: %.2e] "
                        "[gpu: %.1f ms]"
                        "[wall: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            input_var_meter.avg,
                            input_var_min_meter.avg,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            gpu_time_meter.avg,
                            wall_time_meter.avg,
                        )
                    )

                    if optim_stats is not None:
                        logger.info(
                            "[%d, %5d] first moment: %.2e [%.2e %.2e] second moment: %.2e [%.2e %.2e]"
                            % (
                                epoch + 1,
                                itr,
                                optim_stats.get("exp_avg").avg,
                                optim_stats.get("exp_avg").min,
                                optim_stats.get("exp_avg").max,
                                optim_stats.get("exp_avg_sq").avg,
                                optim_stats.get("exp_avg_sq").min,
                                optim_stats.get("exp_avg_sq").max,
                            )
                        )

                    if grad_stats is not None:
                        logger.info(
                            "[%d, %5d] dec_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e"
                            % (
                                epoch + 1,
                                itr,
                                grad_stats.first_layer,
                                grad_stats.last_layer,
                                grad_stats.min,
                                grad_stats.max,
                                grad_stats.global_norm,
                            )
                        )

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        # -- Save Last
        if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"{tag}-e{epoch}.pth.tar"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
