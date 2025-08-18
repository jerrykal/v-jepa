# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# basic
import sys
import copy
import yaml
import torch
import logging
import warnings

# model
import src.models.predictor as vit_pred
import src.models.vision_transformer as video_vit
from src.models.agents.agent import ActorCriticAgent
from src.models.attentive_pooler import AttentivePooler
from src.models.latent_action import LatentActionEncoder
from src.models.world_models.world_model import WorldModel
from src.models.world_models.action_projector import ActionProjector 
from src.models.world_models.state_decoder import RewardsDecoder, TerminationDecoder
from src.models.utils.multimask import (
    MultiMaskWrapper, PredictorMultiMaskWrapper, LatentActionEncoderMultiMaskWrapper)
from torch.nn.parallel import DistributedDataParallel

# utils
from src.utils.tensors import trunc_normal_
from src.utils.replay_buffer import ReplayBuffer
from src.utils.logging import TensorboardLogger
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def init_agent():
    return ActorCriticAgent(

    )

def init_replay_buffer(
        device,
        obs_h, obs_w, obs_c, action_dims, num_envs, 
        max_length=int(1E5), warmup_length=50000, frame_skip=4,
        store_on_gpu=True,
    ):
    return ReplayBuffer(
        obs_shape=(obs_h, obs_w, obs_c),
        action_dims=action_dims,
        num_envs=num_envs,
        max_length=max_length,
        warmup_length=warmup_length,
        frame_skip=frame_skip,
        store_on_gpu=store_on_gpu,
        device=device
    )

def init_world_model(
        device,
        video_model_params:dict,
        latent_action_enc_params:dict,
        state_decoder_params:dict,
        action_projector_params:dict,
        optimizer_params:dict,
        tensorlogger:TensorboardLogger,
        pretrained_model_path=None,
        fine_tune=False,
        use_amp=False,
        amp_dtype=torch.float16,
        **kwargs,
    ):
    '''
    Initialize all components of the world model, including:

    - [JEPA Context Encoder]: Pretrained JEPA video encoder
    - [JEPA Target Encoder]: Pretrained JEPA video encoder
    - [Predictor]: Pretrained model for future latent prediction
    - [Latent Action Encoder]: Pretrained model to encode actions into latent space
    - [Rewards Decoder]: Train from scratch
    - [Termination Decoder]: Train from scratch (predicts termination flags)
    - [Action Projector]: Train from scratch

    Notes:
    - The context and target encoders, predictor, and latent action encoder are loaded from pretrained weights.
    - The rewards decoder, termination decoder, and action projector are initialized from scratch without loading any pretrained weights.
    '''

    encoder, predictor = init_video_model(
        device=device,
        patch_size=video_model_params["patch_size"],
        num_frames=video_model_params["num_frames"],
        tubelet_size=video_model_params["tubelet_size"],
        model_name=video_model_params["model_name"],
        crop_size=video_model_params["crop_size"],
        pred_depth=video_model_params["pred_depth"],
        pred_embed_dim=video_model_params["pred_embed_dim"],
        uniform_power=video_model_params["uniform_power"],
        use_mask_tokens=video_model_params["use_mask_tokens"],
        num_mask_tokens=1,
        zero_init_mask_tokens=False,
        use_sdpa=False,
        adapter_type=video_model_params["adapter_type"],
    )
    target_encoder = copy.deepcopy(encoder)

    latent_action_enc = init_latent_action_encoder(    
        device=device,
        input_dims=encoder.backbone.embed_dim,
        num_heads=latent_action_enc_params["num_heads"],
        d_codebook=latent_action_enc_params["d_codebook"],
        n_codebook=latent_action_enc_params["n_codebook"],
        vq_bias=latent_action_enc_params["vq_bias"],
        vq_commit_weight=latent_action_enc_params["vq_commit_weight"],
        vq_entropy_weight=latent_action_enc_params["vq_entropy_weight"],
        vq_diversity_weight=latent_action_enc_params["vq_diversity_weight"],
    )

    state_pooler, reward_decoder, termin_decoder = init_state_decoder(
        device=device,
        input_dim=encoder.backbone.embed_dim,

        # >> pooler
        num_queries=state_decoder_params["pooler_num_queries"],
        num_heads=state_decoder_params["pooler_num_heads"],
        mlp_ratio=state_decoder_params["mlp_ratio"],
        pooler_depth=state_decoder_params["pooler_depth"],
        norm_layer=state_decoder_params["norm_layer"],
        init_std=state_decoder_params["init_std"],
        qkv_bias=state_decoder_params["qkv_bias"],
        complete_block=state_decoder_params["complete_block"],

        # >> rewards decoder
        reward_hidden_dim=state_decoder_params["reward_hidden_dim"],
        reward_depth=state_decoder_params["reward_depth"],
        num_classes=state_decoder_params["reward_num_classes"],

        # >> termination decoder
        termin_hidden_dim=state_decoder_params["termin_hidden_dim"],
        termin_depth=state_decoder_params["termin_depth"],
    )
    action_projector = init_action_projector(
        device=device,
        hidden_dims=action_projector_params["hidden_dims"], 
        depth=action_projector_params["depth"], 
        quant=latent_action_enc.backbone.quant, 
    )
    if not device == "cpu":
        action_projector = DistributedDataParallel(action_projector)
        state_pooler = DistributedDataParallel(state_pooler)
        reward_decoder = DistributedDataParallel(reward_decoder)
        termin_decoder = DistributedDataParallel(termin_decoder)
        encoder = DistributedDataParallel(encoder)
        target_encoder = DistributedDataParallel(target_encoder)
        predictor = DistributedDataParallel(predictor)
        latent_action_enc = DistributedDataParallel(latent_action_enc)

    if pretrained_model_path is not None:
        (
            encoder,
            target_encoder,
            predictor,
            latent_action_enc
        ) = load_pretrained_model(
            model_path=pretrained_model_path,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            latent_action_enc=latent_action_enc,
            gradient=fine_tune, # If fine-tuning, propagate to each model
            use_ddp=(not device == "cpu")
        )

    if pretrained_model_path is not None and not fine_tune:
        opt_models = [action_projector, state_pooler, reward_decoder, termin_decoder]
        for model in [encoder, target_encoder, predictor, latent_action_enc]:
            model.eval()
    else:
        opt_models = [
            encoder,
            predictor,
            latent_action_enc,
            action_projector,
            state_pooler,
            reward_decoder,
            termin_decoder,
        ]
    
    optimizer, scaler, lr_scheduler, wd_scheduler = init_opt(
        models=opt_models,
        start_lr=optimizer_params["start_lr"],
        ref_lr=optimizer_params["ref_lr"],
        warmup_ratio=optimizer_params["warmup_ratio"],
        wd=optimizer_params["wd"],
        final_wd=optimizer_params["final_wd"],
        final_lr=optimizer_params["final_lr"],
        mixed_precision=optimizer_params["mixed_precision"],
        total_steps=optimizer_params["total_steps"],
        betas=optimizer_params["betas"],
        eps=optimizer_params["eps"],
        zero_init_bias_wd=optimizer_params["zero_init_bias_wd"],
    )
    
    return WorldModel(
        context_encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        latent_action_encoder=latent_action_enc,
        state_pooler=state_pooler,
        rewards_decoder=reward_decoder,
        termination_decoder=termin_decoder,
        action_projector=action_projector,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=lr_scheduler,
        wd_scheduler=wd_scheduler,
        tb_logger=tensorlogger,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        **kwargs
    )

# -- init function
def _init_weights(m):
    if isinstance(m, torch.nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_action_projector(
    device, 
    quant,
    hidden_dims: int=1024, 
    depth: int=2, 
):
    action_projector = ActionProjector(
        hidden_dims=hidden_dims, 
        depth=depth, 
        quant=quant
    )
    for m in action_projector.modules():
        _init_weights(m)
    action_projector = action_projector.to(device)


    logger.info(action_projector)
    logger.info(f'Rewards Decoder number of parameters: {_count_parameters(action_projector)}')

    return action_projector

def init_state_decoder(
    device,
    input_dim: int=384,

    # >> pooler
    num_queries: int=1,
    num_heads: int=4,
    mlp_ratio: int=4,
    pooler_depth: int=3,
    norm_layer: torch.nn.Module=torch.nn.LayerNorm,
    init_std: float=0.02,
    qkv_bias: bool=True,
    complete_block: bool=True,

    # >> rewards decoder
    reward_hidden_dim: int=256,
    reward_depth: int=2,
    num_classes: int=255,

    # >> termination decoder
    termin_hidden_dim: int=2560,
    termin_depth: int=2,
):
    pooler = AttentivePooler(
        num_queries=num_queries,
        embed_dim=input_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        depth=pooler_depth,
        norm_layer=norm_layer,
        init_std=init_std,
        qkv_bias=qkv_bias,
        complete_block=complete_block,
    )

    reward_decoder = RewardsDecoder(
        num_classes=num_classes, 
        input_dim=input_dim, 
        hidden_dim=reward_hidden_dim, 
        depth=reward_depth)
    
    termin_decoder = TerminationDecoder(
        input_dim=input_dim, 
        hidden_dim=termin_hidden_dim, 
        depth=termin_depth)
    
    for m in reward_decoder.modules():
        _init_weights(m)
        
    for m in termin_decoder.modules():
        _init_weights(m)

    pooler = pooler.to(device)
    reward_decoder = reward_decoder.to(device)
    termin_decoder = termin_decoder.to(device)

    logger.info(pooler)
    logger.info(reward_decoder)
    logger.info(termin_decoder)
    logger.info(f'State pooler of parameters: {_count_parameters(pooler)}')
    logger.info(f'Rewards Decoder number of parameters: {_count_parameters(reward_decoder)}')
    logger.info(f'Termination Decoder number of parameters: {_count_parameters(termin_decoder)}')
    return pooler, reward_decoder, termin_decoder
    
def init_latent_action_encoder(
    device,
    input_dims: int = 192, 
    num_heads: int = 8,
    d_codebook: int = 10,
    n_codebook: int = 1,
    vq_bias: bool = True,
    vq_commit_weight: float = 0.25,
    vq_entropy_weight: float = 0.1,
    vq_diversity_weight: float = 1.,
    ):
    la_enc = LatentActionEncoder(
        input_dims=input_dims, 
        num_heads=num_heads,
        d_codebook=d_codebook,
        n_codebook=n_codebook,
        vq_bias=vq_bias,
        vq_commit_weight=vq_commit_weight,
        vq_entropy_weight=vq_entropy_weight,
        vq_diversity_weight=vq_diversity_weight,
        quant_loss_weight=1.0,
    )
    la_enc = LatentActionEncoderMultiMaskWrapper(la_enc)
    
    for m in la_enc.modules():
        _init_weights(m)

    la_enc = la_enc.to(device)

    logger.info(la_enc)
    logger.info(f'Predictor number of parameters: {_count_parameters(la_enc)}')
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
    use_sdpa=False, # not using
    adapter_type="None",
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa, # not using
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


    for m in encoder.modules():
        _init_weights(m)

    for m in predictor.modules():
        _init_weights(m)

    encoder.to(device)
    predictor.to(device)

    logger.info(encoder)
    logger.info(predictor)
    logger.info(f'Encoder number of parameters: {_count_parameters(encoder)}')
    logger.info(f'Predictor number of parameters: {_count_parameters(predictor)}')

    return encoder, predictor

def init_opt(
    models, 
    start_lr,
    ref_lr,
    warmup_ratio,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    total_steps=1e7,
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


    lr_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(total_steps * warmup_ratio),
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

    return optimizer, scaler, lr_scheduler, wd_scheduler

# -- Load Function
def _strip_module_prefix(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def _load_component(checkpoint, name, model, use_ddp):
    if model is not None and name in checkpoint:
        try:
            ckpt= checkpoint[name] if use_ddp else _strip_module_prefix(checkpoint[name])
            msg = model.load_state_dict(ckpt)
            logger.info(f'Loaded {name} with msg: {msg}')
        except Exception as e:
            logger.warning(f'Failed to load {name}: {e}')
    else:
        logger.warning(f'No "{name}" found in checkpoint.')
    return model

def load_pretrained_model(
    model_path,
    encoder,
    target_encoder,
    predictor,
    latent_action_enc,
    gradient=False,
    use_ddp=True
):
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint: {e}')
        return encoder, target_encoder, predictor, latent_action_enc

    try:
        for name, module in [
            ('encoder', encoder),
            ('target_encoder', target_encoder),
            ('predictor', predictor),
            ('latent_action_encoder', latent_action_enc)
        ]:
            module = _load_component(checkpoint, name, module, use_ddp)
            if not gradient:
                for param in module.parameters():
                    param.requires_grad_(False)

    except Exception as e:
        logger.info(f'Failed to load models from checkpoint: {e}')

    return encoder, target_encoder, predictor, latent_action_enc

    
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
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
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
