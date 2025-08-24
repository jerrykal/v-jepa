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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from app.vit_decoder.utils import (
    init_models,
    init_opt,
    load_checkpoint,
    load_jepa_encoder,
    patchify,
)
from app.vjepa.transforms import make_transforms
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
    eval_freq = cfgs_meta.get("eval_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
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
    decoder_depth = cfgs_model.get("decoder_depth", 8)
    decoder_num_heads = cfgs_model.get("decoder_num_heads", 16)
    decoder_mlp_ratio = cfgs_model.get("decoder_mlp_ratio", 4.0)

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    train_dataset_paths = cfgs_data.get("train_datasets", [])
    train_datasets_weights = cfgs_data.get("train_datasets_weights", None)
    if train_datasets_weights is not None:
        assert len(train_datasets_weights) == len(train_dataset_paths), (
            "Must have one sampling weight specified for each dataset"
        )
    eval_dataset_paths = cfgs_data.get("eval_datasets", [])
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
    train_log_file = os.path.join(folder, f"{tag}_r{rank}_train.csv")
    eval_log_file = os.path.join(folder, f"{tag}_r{rank}_eval.csv")
    latest_file = f"{tag}-latest.pth.tar"
    latest_path = os.path.join(folder, latest_file)
    best_file = f"{tag}-best.pth.tar"
    best_path = os.path.join(folder, best_file)
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    train_csv_logger = CSVLogger(
        train_log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "grad-norm"),
        ("%d", "gpu-time(ms)"),
        ("%d", "wall-time(ms)"),
    )
    if eval_freq > 0:
        eval_csv_logger = CSVLogger(
            eval_log_file,
            ("%d", "epoch"),
            ("%.5f", "loss"),
        )

    # -- init model
    encoder, decoder = init_models(
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        img_size=crop_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        decoder_mlp_ratio=decoder_mlp_ratio,
        decoder_norm_layer=nn.LayerNorm,
    )

    # -- make data transforms
    train_transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )
    short_side_size = int(crop_size * 256 / 224)
    normalize_mean = torch.tensor([0.485, 0.456, 0.406])
    normalize_std = torch.tensor([0.229, 0.224, 0.225])
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(crop_size, crop_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )

    # -- init data-loaders/samplers
    train_loader, train_sampler = init_data(
        data=dataset_type,
        root_path=train_dataset_paths,
        batch_size=batch_size,
        training=True,
        clip_len=num_frames,
        frame_sample_rate=sampling_rate,
        filter_short_videos=filter_short_videos,
        decode_one_clip=decode_one_clip,
        duration=duration,
        num_clips=num_clips,
        transform=train_transform,
        datasets_weights=train_datasets_weights,
        collator=None,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        rank=rank,
        log_dir=folder if log_resource_util_data else None,
    )
    if eval_freq > 0:
        eval_loader, _ = init_data(
            data=dataset_type,
            root_path=eval_dataset_paths,
            batch_size=batch_size,
            training=False,
            clip_len=num_frames,
            frame_sample_rate=sampling_rate,
            filter_short_videos=filter_short_videos,
            decode_one_clip=decode_one_clip,
            duration=duration,
            num_clips=num_clips,
            transform=eval_transform,
            collator=None,
            num_workers=num_workers,
            world_size=world_size,
            pin_mem=pin_mem,
            rank=rank,
            log_dir=folder if log_resource_util_data else None,
        )

    try:
        _dlen = len(train_loader)
    except Exception:  # Different interface for webdataset
        _dlen = train_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataest length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler
    optimizer, scaler, lr_scheduler = init_opt(
        models=[encoder, decoder],
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
        p.requires_grad = False

    start_epoch = 0

    # -- load pre-trained encoder
    assert pre_train_model is not None, "Pre-trained encoder must be provided"
    load_jepa_encoder(pre_train_model, encoder)

    # -- load training checkpoint
    if load_model:
        epoch = load_checkpoint(
            r_path=load_path,
            decoder=decoder,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            lr_scheduler.step()

    # -- wrap decoder with DDP
    decoder = DistributedDataParallel(decoder, static_graph=True)

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "decoder": decoder.state_dict(),
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
    loader = iter(train_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        train_sampler.set_epoch(start_epoch)
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                udata = next(loader)
            except Exception:
                loader = iter(train_loader)
                udata = next(loader)

    # -- TRAINING LOOP
    best_eval_loss = float("inf")
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        # -- update distributed-data-loader epoch
        train_sampler.set_epoch(epoch)

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
                loader = iter(train_loader)
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
                    # Encode video clips
                    jepa_features = encoder(clips)
                    jepa_features = F.layer_norm(
                        jepa_features, (jepa_features.size(-1),)
                    )

                    # Reconstruct video clips from encoded representations
                    pred = decoder(jepa_features)

                    # Patchify video clips to obtain target patches
                    target = patchify(clips, patch_size, tubelet_size)

                    # Compute the loss
                    loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

                # Step 2. Backward & step
                _dec_norm = 0.0
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if clip_grad is not None:
                    _dec_norm = nn.utils.clip_grad_norm_(
                        decoder.parameters(), clip_grad
                    )
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                grad_stats = grad_logger(decoder.named_parameters())
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
                train_csv_logger.log(
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
        logger.info("avg. loss %.3f" % loss_meter.avg)

        # -- EVALUATION
        if eval_freq > 0 and (epoch + 1) % eval_freq == 0:
            logger.info("Evaluating...")
            eval_loss_meter = AverageMeter()

            encoder.eval()
            decoder.eval()

            with torch.no_grad():
                for udata in eval_loader:
                    clips = torch.cat(
                        [u.to(device, non_blocking=True) for u in udata[0]], dim=0
                    )
                    jepa_features = encoder(clips)
                    jepa_features = F.layer_norm(
                        jepa_features, (jepa_features.size(-1),)
                    )
                    pred = decoder(jepa_features)
                    target = patchify(clips, patch_size, tubelet_size)
                    loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

                    eval_loss_meter.update(loss)

            # Log evaluation stats
            eval_csv_logger.log(
                epoch + 1,
                eval_loss_meter.avg,
            )
            logger.info(f"Evaluation loss: {eval_loss_meter.avg:.3f}")

            # Saving best checkpoint
            if eval_loss_meter.avg < best_eval_loss:
                logger.info(
                    f"New best evaluation loss: {eval_loss_meter.avg:.3f}, saving..."
                )
                best_eval_loss = eval_loss_meter.avg
                save_checkpoint(epoch + 1, best_path)

            encoder.train()
            decoder.train()

        # -- Save checkpoint
        if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                save_every_file = f"{tag}-e{epoch + 1}.pth.tar"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
