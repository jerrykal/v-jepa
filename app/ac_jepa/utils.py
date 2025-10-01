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
import torch.nn as nn

from typing import Dict, List

import src.models.vision_transformer as video_vit
import src.models.predictor as vit_pred
from src.models.latent_action import LatentActionEncoder
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper, LatentActionEncoderMultiMaskWrapper
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_
from src.utils.saveloader import load_component

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def _is_in_training(name: str, training_list: List[str]) -> bool:
    for t in training_list:
        if name == t or name.startswith(t + "."):
            return True
    return False

def build_load_model_dict(
    all_named_modules: Dict[str, nn.Module],
    training_model_list: List[str],
    trainable: bool = False,
) -> Dict[str, nn.Module]:
    unknown = [t for t in training_model_list
               if not any(t == k or k.startswith(t + ".") for k in all_named_modules.keys())]
    if unknown:
        logger.warning(f"Unknown training names (ignored): {unknown}")

    def _is_in_training(name: str) -> bool:
        return any(name == t or name.startswith(t + ".") for t in training_model_list)

    if trainable:
        load_dict = {name: module for name, module in all_named_modules.items() if _is_in_training(name)}
        logger.info(f"Selected trainable modules: {list(load_dict.keys())}")
    else:
        load_dict = {name: module for name, module in all_named_modules.items() if not _is_in_training(name)}
        logger.info(f"Selected frozen (not trainable) modules: {list(load_dict.keys())}")

    if not load_dict:
        logger.warning("No modules matched the selection.")

    return load_dict


def load_pretrained_model(
    model_path: str,
    load_model_dict: dict[str, nn.Module],
    *,
    use_ddp: bool = False,
    trainable: bool = False,
):

    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint: {e}')
        return load_model_dict

    for name, module in load_model_dict.items():
        try:
            module = load_component(checkpoint, name, module, use_ddp)
            for param in module.parameters():
                param.requires_grad = trainable
        except Exception as e:
            logger.warning(f'[{name}] failed to load: {e}')

    return load_model_dict

def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    latent_action_enc,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'), weights_only=True)
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')

    epoch = 0
    try:
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(
                f'loaded pretrained target encoder from epoch {epoch} with msg: {msg}'
            )
        # -- loading latent_action_encoder
        if latent_action_enc is not None and 'latent_action_encoder' in checkpoint:
            pretrained_dict = checkpoint['latent_action_encoder']
            msg = latent_action_enc.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained latent_action_encoder from epoch {epoch} with msg: {msg}')
        else:
            logger.warning('latent_action_encoder not found in checkpoint or model is None.')

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
        encoder,
        predictor,
        target_encoder,
        latent_action_enc,
        opt,
        scaler,
        epoch,
    )

def init_latent_action_encoder(
    device,
    inp_dims: int = 192, 
    num_heads: int = 8,
    d_codebook: int = 10,
    n_codebook: int = 1,
    vq_bias: bool = True,
    vq_commit_weight: float = 0.25,
    vq_entropy_weight: float = 0.1,
    vq_diversity_weight: float = 1.,
    ):
    la_enc = LatentActionEncoder(
        input_dims=inp_dims, 
        num_heads=num_heads,
        d_codebook=d_codebook,
        n_codebook=n_codebook,
        vq_bias=vq_bias,
        vq_commit_weight=vq_commit_weight,
        vq_entropy_weight=vq_entropy_weight,
        vq_diversity_weight=vq_diversity_weight,
    )
    la_enc = LatentActionEncoderMultiMaskWrapper(la_enc)
    
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in la_enc.modules():
        init_weights(m)

    la_enc = la_enc.to(device)
    logger.info(la_enc)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Predictor number of parameters: {count_parameters(la_enc)}')
    return la_enc

def init_video_model(
    device,
    patch_size=16,
    num_frames=16,
    tubelet_size=2,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    adapter_type="None",
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
    )
    encoder = MultiMaskWrapper(encoder)
    predictor = vit_pred.__dict__['vit_predictor'](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa,
        adapter_type=adapter_type,
        action_dim=encoder.backbone.embed_dim,
    )
    predictor = PredictorMultiMaskWrapper(predictor)

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

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f'Encoder number of parameters: {count_parameters(encoder)}')
    logger.info(f'Predictor number of parameters: {count_parameters(predictor)}')

    return encoder, predictor

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