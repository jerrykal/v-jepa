# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys
import warnings
import yaml


import torch

import src.models.vision_transformer as video_vit
from src.models.decoder  import VideoDecoder
from src.models.utils.multimask import MultiMaskWrapper
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def load_jepa_encoder(
    model_path,
    target_encoder,
):
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint: {e}')
        return target_encoder

    try:
        # -- loading target_encoder
        if target_encoder is not None and 'target_encoder' in checkpoint:
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(f'Loaded JEPA target_encoder with msg: {msg}')
        else:
            logger.warning('No "target_encoder" found in checkpoint.')


    except Exception as e:
        logger.info(f'Failed to load JEPA encoder/target_encoder: {e}')

    return target_encoder

def load_checkpoint(
    r_path,
    decoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')

    epoch = 0
    try:
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['decoder']
        msg = decoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained decoder from epoch {epoch} with msg: {msg}')


        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return (
        decoder,
        opt,
        scaler,
        epoch,
    )

def init_video_model(
    device,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    model_name='vit_base',
    crop_size=224,
    decoder_layers=4,
    decoder_stem_dim=64,
    uniform_power=False,
    use_sdpa=False,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
    )

    decoder = VideoDecoder(
        embed_dim=encoder.embed_dim,
        stem_dim=decoder_stem_dim,
        num_layers=decoder_layers,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        height=crop_size,
        width=crop_size,
        num_frames=num_frames,
    )
    encoder = MultiMaskWrapper(encoder)
    decoder = MultiMaskWrapper(decoder)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)


    encoder.to(device)
    decoder.to(device)

    logger.info(encoder)
    logger.info(decoder)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f'Encoder number of parameters: {count_parameters(encoder)}')
    logger.info(f'Decoder number of parameters: {count_parameters(decoder)}')
    return encoder, decoder

def init_opt(
    models, 
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = []

    for model in models:
        param_groups.append({
            'params': (p for n, p in model.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        })
        param_groups.append({
            'params': (p for n, p in model.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': zero_init_bias_wd,
            'weight_decay': 0,
        })

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    total_steps = int(ipe_scale * num_epochs * iterations_per_epoch)

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=total_steps,
    )

    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=total_steps,
    )

    scaler = torch.amp.GradScaler(device='cuda') if mixed_precision else None

    return optimizer, scaler, scheduler, wd_scheduler