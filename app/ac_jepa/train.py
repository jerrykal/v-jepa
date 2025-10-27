# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from app.ac_jepa.transforms import make_transforms
from app.ac_jepa.utils import (
    build_load_model_dict,
    init_latent_action_encoder,
    init_opt,
    init_video_model,
    load_checkpoint,
    load_pretrained_model,
)
from einops import rearrange
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
from torch.nn.parallel import DistributedDataParallel

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
    which_dtype = cfgs_meta.get("dtype")
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
    pre_train_model = cfgs_meta.get("pre_train_model", None)
    second_stage = cfgs_meta.get("second_stage", False)

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", True)
    la_enc_num_heads = cfgs_model.get("latent_action_num_heads", 8)
    la_d_codebook = cfgs_model.get("dims_aciton_codebook", 10)
    la_n_codebook = cfgs_model.get("number_aciton_codebook", 1)
    vq_bias = cfgs_model.get("vq_bias", True)
    vq_commit_weight = cfgs_model.get("vq_commit_weight", 0.25)
    vq_entropy_weight = cfgs_model.get("vq_entropy_weight", 0.1)
    vq_diversity_weight = cfgs_model.get("vq_diversity_weight", 1.0)
    training_model_list = cfgs_model.get("training_model_list", ["encoder", "ac_predictor", "latent_action_enc"])

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights", None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), "Must have one sampling weight specified for each dataset"
    batch_size = cfgs_data.get("batch_size")
    num_clips = cfgs_data.get("num_clips")
    num_frames = cfgs_data.get("num_frames")
    tubelet_size = cfgs_data.get("tubelet_size")
    sampling_rate = cfgs_data.get("sampling_rate")
    duration = cfgs_data.get("clip_duration", None)
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    num_patches_per_frame = (crop_size // patch_size) ** 2
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

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    reg_coeff = cfgs_loss.get("reg_coeff")
    quant_coeff = cfgs_loss.get("quant_coeff")

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    clip_grad = cfgs_opt.get("clip_grad", None)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
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
        ("%.5f", "loss-jepa"),
        ("%.5f", "loss-quant"),
        ("%.5f", "reg-loss"),
        ("%.5f", "enc-grad-norm"),
        ("%.5f", "pred-grad-norm"),
        ("%.5f", "la-grad-norm"),
        ("%d", "gpu-time(ms)"),
        ("%d", "wall-time(ms)"),
    )

    # -- init model
    encoder, ac_predictor = init_video_model(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
    )
    target_encoder = copy.deepcopy(encoder)

    # -- init latent action model
    latent_action_enc = init_latent_action_encoder(
        device=device,
        input_dim=encoder.backbone.embed_dim,
        num_patches_per_frame=num_patches_per_frame,
        num_heads=la_enc_num_heads,
        d_codebook=la_d_codebook,
        n_codebook=la_n_codebook,
        vq_bias=vq_bias,
        vq_commit_weight=vq_commit_weight,
        vq_entropy_weight=vq_entropy_weight,
        vq_diversity_weight=vq_diversity_weight,
        use_sdpa=use_sdpa,
    )

    all_named_modules = {
        "encoder": encoder,
        "target_encoder": target_encoder,
        "ac_predictor": ac_predictor,
        "latent_action_encoder": latent_action_enc,
    }

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
    training_modules = [all_named_modules[name] for name in training_model_list if name in all_named_modules]
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        models=training_modules,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )

    ac_predictor = DistributedDataParallel(ac_predictor, static_graph=True)
    latent_action_enc = DistributedDataParallel(latent_action_enc, static_graph=True)
    encoder = DistributedDataParallel(encoder)
    target_encoder = DistributedDataParallel(target_encoder)

    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0
    # -- load training checkpoint
    if pre_train_model:
        logger.info(f"Load pretrained checkpoint:{pre_train_model}")
        _ = load_pretrained_model(
            pre_train_model,
            build_load_model_dict(
                all_named_modules=all_named_modules, training_model_list=training_model_list, trainable=False
            ),
            use_ddp=False,
            trainable=False,
        )

        if second_stage:
            logger.info("Training second stage for encoder!")
            _ = load_pretrained_model(
                pre_train_model,
                build_load_model_dict(
                    all_named_modules=all_named_modules, training_model_list=training_model_list, trainable=True
                ),
                use_ddp=False,
                trainable=True,
            )

    if load_model or os.path.exists(latest_path):
        (
            encoder,
            ac_predictor,
            target_encoder,
            latent_action_enc,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            ac_predictor=ac_predictor,
            target_encoder=target_encoder,
            latent_action_enc=latent_action_enc,
            opt=optimizer,
            scaler=scaler,
        )

        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "ac_predictor": ac_predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "latent_action_encoder": latent_action_enc.state_dict(),
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
        logger.info(f"Epoch {epoch + 1}")

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        input_var_meter = AverageMeter()
        input_var_min_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()
        quant_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
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

            def load_clips(udata=udata):
                # -- unsupervised video clips
                # Put each clip on the GPU and concatenate along batch
                # dimension
                clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)

                return clips

            clips = load_clips()

            # Rearrange clips from (B, C, T, H, W) to (B * T, C, 2, H, W),
            # flattening the temporal dimension into the batch so each frame can
            # be processed independently as an image. This enables the JEPA encoder
            # to operate over frames individually.
            clips = rearrange(clips, "b c t h w -> (b t) c 1 h w").repeat(1, 1, 2, 1, 1)

            def train_step(epoch=epoch, clips=clips):
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_targets(clips):
                    with torch.no_grad():
                        # Encode clips, then merge temporal dimension back with patch dimension
                        targets = target_encoder(clips)
                        targets = rearrange(targets, "(b t) p d -> b t p d", t=num_frames)
                        targets = F.layer_norm(targets, (targets.size(-1),))  # normalize over feature-dim  [B, N, D]

                        return targets

                def forward_actions(targets):
                    acts, loss_quant = latent_action_enc(targets)
                    return acts, loss_quant

                def forward_predictions(targets, acts):
                    preds = ac_predictor(targets, acts)
                    preds = rearrange(preds, "b (t p) d -> b t p d", t=num_frames)
                    return preds

                def loss_fn(preds, targets):
                    preds_shifted = preds[:, :-1]
                    targets_shifted = targets[:, 1:]
                    loss = torch.mean(torch.abs(preds_shifted - targets_shifted) ** loss_exp) / loss_exp
                    return loss

                def reg_fn(z):
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)

                # Step 1. Forward
                loss_jepa, loss_reg = 0.0, 0.0
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    targets = forward_targets(clips)
                    acts, loss_quant = forward_actions(targets)
                    preds = forward_predictions(targets, acts)

                    loss_jepa = loss_fn(preds, targets)  # jepa prediction loss
                    pstd_pred = reg_fn(preds)  # predictor variance across patches
                    loss_reg += torch.mean(F.relu(1.0 - pstd_pred))

                loss = loss_jepa + reg_coeff * loss_reg + loss_quant * quant_coeff

                # Step 2. Backward & step
                _enc_norm, _pred_norm, _la_norm = 0.0, 0.0, 0.0
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if (epoch > warmup) and (clip_grad is not None):
                    _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    _pred_norm = torch.nn.utils.clip_grad_norm_(ac_predictor.parameters(), clip_grad)
                    _la_norm = torch.nn.utils.clip_grad_norm_(latent_action_enc.parameters(), clip_grad)
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                grad_stats.global_norm = float(_enc_norm)
                grad_stats_pred = grad_logger(ac_predictor.named_parameters())
                grad_stats_pred.global_norm = float(_pred_norm)
                grad_stats_la_enc = grad_logger(latent_action_enc.named_parameters())
                grad_stats_la_enc.global_norm = float(_la_norm)
                optimizer.zero_grad()
                optim_stats = adamw_logger(optimizer)

                # Step 3. momentum update of target encoder
                if "target_encoder" in training_model_list:
                    m = next(momentum_scheduler)
                    with torch.no_grad():
                        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters(), strict=True):
                            param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                return (
                    float(loss),
                    float(loss_jepa),
                    float(loss_quant),
                    float(loss_reg),
                    _new_lr,
                    _new_wd,
                    grad_stats,
                    grad_stats_pred,
                    grad_stats_la_enc,
                    optim_stats,
                )

            (
                (
                    loss,
                    loss_jepa,
                    loss_quant,
                    loss_reg,
                    _new_lr,
                    _new_wd,
                    grad_stats,
                    grad_stats_pred,
                    grad_stats_la_enc,
                    optim_stats,
                ),
                gpu_etime_ms,
            ) = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            input_var = float(AllReduce.apply(clips.view(clips.shape[0], -1).var(dim=1).mean(dim=0)))
            input_var_min = float(AllReduce.apply(torch.min(clips.view(clips.shape[0], -1).var(dim=1))))
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
            jepa_loss_meter.update(loss_jepa)
            quant_loss_meter.update(loss_quant)
            reg_loss_meter.update(loss_reg)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)

            # -- Logging
            csv_logger.log(
                epoch + 1,
                itr,
                loss,
                loss_jepa,
                loss_quant,
                loss_reg,
                grad_stats.global_norm,
                grad_stats_pred.global_norm,
                grad_stats_la_enc.global_norm,
                gpu_etime_ms,
                iter_elapsed_time_ms,
            )
            if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                logger.info(
                    f"[{epoch + 1},{itr:5d}] loss: {loss_meter.avg:.3f} | p: {jepa_loss_meter.avg:.3f} "
                    f"q: {quant_loss_meter.avg:.3f} r: {reg_loss_meter.avg:.3f} | "
                    f"var: {input_var_meter.avg:.3f} {input_var_min_meter.avg:.3f} | "
                    f"[wd: {_new_wd:.2e}] [lr: {_new_lr:.2e}] "
                    f"[mem: {torch.cuda.max_memory_allocated() / 1024.0**2:.2e}] "
                    f"[gpu: {gpu_time_meter.avg:.1f}ms] [wall: {wall_time_meter.avg:.1f}ms]"
                )

                if optim_stats is not None:
                    logger.info(
                        f"[{epoch + 1},{itr:5d}] 1st: {optim_stats.get('exp_avg').avg:.2e} "
                        f"[{optim_stats.get('exp_avg').min:.2e},{optim_stats.get('exp_avg').max:.2e}] "
                        f"2nd: {optim_stats.get('exp_avg_sq').avg:.2e} "
                        f"[{optim_stats.get('exp_avg_sq').min:.2e},{optim_stats.get('exp_avg_sq').max:.2e}]"
                    )

                if grad_stats is not None:
                    logger.info(
                        f"[{epoch + 1},{itr:5d}] enc_grad: fl[{grad_stats.first_layer:.2e},{grad_stats.last_layer:.2e}] "
                        f"mn/mx({grad_stats.min:.2e},{grad_stats.max:.2e}) {grad_stats.global_norm:.2e}"
                    )

                if grad_stats_pred is not None:
                    logger.info(
                        f"[{epoch + 1},{itr:5d}] pred_grad: fl[{grad_stats_pred.first_layer:.2e},{grad_stats_pred.last_layer:.2e}] "
                        f"mn/mx({grad_stats_pred.min:.2e},{grad_stats_pred.max:.2e}) {grad_stats_pred.global_norm:.2e}"
                    )

                if grad_stats_la_enc is not None:
                    logger.info(
                        f"[{epoch + 1},{itr:5d}] la_grad: fl[{grad_stats_la_enc.first_layer:.2e},{grad_stats_la_enc.last_layer:.2e}] "
                        f"mn/mx({grad_stats_la_enc.min:.2e},{grad_stats_la_enc.max:.2e}) {grad_stats_la_enc.global_norm:.2e}"
                    )

            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint
        logger.info(f"avg. loss {loss_meter.avg:.3f}")
        # -- Save Last
        if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"{tag}-e{epoch}.pth.tar"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
