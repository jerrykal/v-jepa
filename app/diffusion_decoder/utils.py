# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDIMScheduler, EDMEulerScheduler
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

import src.models.vision_transformer as video_vit
from src.models.diffusion_decoder import get_unet_and_scheduler
from src.models.utils.multimask import MultiMaskWrapper
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_jepa_encoder(
    model_path: str,
    encoder: nn.Module,
):
    try:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint: {e}")
        return encoder

    try:
        # -- loading target_encoder
        if encoder is not None and "encoder" in checkpoint:
            pretrained_dict = checkpoint["encoder"]

            # NOTE: since the pre-trained weights are saved with DDP wrapper, we need to remove the DDP prefix "module." if we are not using DDP
            if not isinstance(encoder, DistributedDataParallel):
                pretrained_dict = {
                    k.replace("module.", ""): v for k, v in pretrained_dict.items()
                }

            msg = encoder.load_state_dict(pretrained_dict)
            logger.info(f"Loaded JEPA encoder with msg: {msg}")
        else:
            logger.warning('No "encoder" found in checkpoint.')

    except Exception as e:
        logger.info(f"Failed to load JEPA encoder: {e}")


def get_pretrained_vae(
    model_id: str,
    device: torch.device,
):
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    vae.to(device)
    logger.info(f"Loaded VAE from {model_id}")

    return vae


def load_checkpoint(
    r_path: str,
    unet: nn.Module,
    noise_scheduler: DDIMScheduler,
    opt: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> int:
    try:
        checkpoint = torch.load(
            r_path, map_location=torch.device("cpu"), weights_only=False
        )
    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")

    epoch = 0
    try:
        epoch = checkpoint["epoch"]

        # -- loading denoising unet
        unet.register_to_config(**checkpoint["unet_config"])

        # NOTE: since the pre-trained weights are saved with DDP wrapper, we need to remove the DDP prefix "module." if we are not using DDP
        if not isinstance(unet, DistributedDataParallel):
            checkpoint["unet"] = {
                k.replace("module.", ""): v for k, v in checkpoint["unet"].items()
            }

        msg = unet.load_state_dict(checkpoint["unet"])
        logger.info(
            f"loaded pretrained denoising UNet from epoch {epoch} with msg: {msg}"
        )

        # -- loading noise scheduler
        noise_scheduler.register_to_config(**checkpoint["noise_scheduler_config"])
        logger.info("loaded noise scheduler configuration")

        # -- loading optimizer
        if opt is not None:
            opt.load_state_dict(checkpoint["opt"])
            if scaler is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            logger.info(f"loaded optimizers from epoch {epoch}")
            logger.info(f"read-path: {r_path}")
        del checkpoint

    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")
        epoch = 0

    return epoch


def init_models(
    device: torch.device,
    patch_size: int = 16,
    num_frames: int = 16,
    tubelet_size: int = 2,
    model_name: str = "vit_base",
    crop_size: int = 224,
    sample_size: int = 224,
    uniform_power: bool = False,
    use_sdpa: bool = False,
    in_channels: int = 3,
    out_channels: int = 3,
    layers_per_block: int = 2,
    attention_head_dim: int = 8,
    dropout: float = 0.0,
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
    down_block_types: tuple[str, ...] = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types: tuple[str, ...] = (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),
    scheduler_beta_start: float = 0.00085,
    scheduler_beta_end: float = 0.012,
    scheduler_beta_schedule: str = "scaled_linear",
    scheduler_prediction_type: str = "epsilon",
    cross_attn_cond: bool = True,
    cross_attention_dim: int | None = None,
    in_concat_cond: bool = False,
    do_edm_style_training: bool = False,
) -> tuple[nn.Module, nn.Module, DDIMScheduler | EDMEulerScheduler]:
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
    )
    encoder = MultiMaskWrapper(encoder)

    if in_concat_cond:
        # Modified the input channel to accommodate concatenated JEPA conditioning.
        in_channels = (
            in_channels + (num_frames // tubelet_size) * encoder.backbone.embed_dim
        )

    if cross_attention_dim is None:
        cross_attention_dim = encoder.backbone.embed_dim

    # Diffusion decoder and noise scheduler
    unet, noise_scheduler = get_unet_and_scheduler(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        attention_head_dim=attention_head_dim,
        dropout=dropout,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        encoder_hid_dim=encoder.backbone.embed_dim,
        cross_attention_dim=cross_attention_dim if cross_attn_cond else None,
        num_class_embeds=num_frames,
        scheduler_beta_start=scheduler_beta_start,
        scheduler_beta_end=scheduler_beta_end,
        scheduler_beta_schedule=scheduler_beta_schedule,
        scheduler_prediction_type=scheduler_prediction_type,
        do_edm_style_training=do_edm_style_training,
    )

    def init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    encoder.to(device)
    unet.to(device)
    logger.info(encoder)
    logger.info(unet)
    logger.info(noise_scheduler)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Denoising UNet number of parameters: {count_parameters(unet)}")

    return encoder, unet, noise_scheduler


def init_opt(
    models: list[nn.Module],
    scheduler_type: str,
    iterations_per_epoch: int,
    lr: float,
    warmup: float,
    num_epochs: int,
    wd: float = 1e-6,
    mixed_precision: bool = False,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    zero_init_bias_wd: bool = True,
) -> tuple[
    torch.optim.AdamW,
    torch.cuda.amp.GradScaler | None,
    LambdaLR,
]:
    param_groups = []
    for model in models:
        param_groups.append(
            {
                "params": (
                    p
                    for n, p in model.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                )
            }
        )
        param_groups.append(
            {
                "params": (
                    p
                    for n, p in model.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
            }
        )

    logger.info("Using AdamW")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        weight_decay=wd,
        eps=eps,
    )
    lr_scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(warmup * iterations_per_epoch),
        num_training_steps=int(num_epochs * iterations_per_epoch),
    )
    scaler = torch.amp.GradScaler(device="cuda") if mixed_precision else None

    return optimizer, scaler, lr_scheduler
