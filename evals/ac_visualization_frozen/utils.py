# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import torch
import torch.nn as nn
import copy
from torch.nn.parallel import DistributedDataParallel

import logging
import yaml

import src.models.vision_transformer as video_vit
import src.models.predictor as vit_pred

from glob import glob
from einops import rearrange
from src.models.latent_action import LatentActionEncoder
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper, LatentActionEncoderMultiMaskWrapper
from src.models.vit_decoder import ViTVideoDecoder
from src.utils.tensors import trunc_normal_

from src.utils.logging import (
    get_logger
)

logger = get_logger(__name__)

def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def _join_if_relative(base, maybe_path):
    if not maybe_path:
        return None
    return maybe_path if os.path.isabs(maybe_path) else os.path.join(base, maybe_path)

def _find_ckpt_file(base_dir):
    """
    Auto-pick a checkpoint under base_dir.
    Priority: filenames containing 'best' > 'final'/'last' > newest mtime.
    """
    patterns = ["*.pth", "*.pth.tar", "*.pt"]
    cands = []
    for pat in patterns:
        cands += glob(os.path.join(base_dir, pat))
    if not cands:
        return None
    def _score(p):
        name = os.path.basename(p).lower()
        pref = 2 if "best" in name else (1 if ("final" in name or "last" in name) else 0)
        return (pref, os.path.getmtime(p))
    return sorted(cands, key=_score, reverse=True)[0]

def merge_eval_args_with_components(args_eval):
    """
    Expect args_eval['meta'] contains:
      - ac_jepa_path: folder containing params-pretrain.yaml and a .pth
      - decoder_path:     folder containing params-pretrain.yaml and a .pth
    This function loads both YAMLs, resolves ckpt paths, and merges back into args_eval.
    """
    meta = args_eval.get("meta", {})
    ac_dir = meta.get("ac_jepa_path")
    dec_dir = meta.get("decoder_path")
    assert ac_dir and dec_dir, "Please set meta.ac_jepa_path and meta.decoder_path in eval.yaml"

    ac_yaml = os.path.join(ac_dir, "params-pretrain.yaml")
    dec_yaml = os.path.join(dec_dir, "params-pretrain.yaml")
    assert os.path.exists(ac_yaml), f"Missing {ac_yaml}"
    assert os.path.exists(dec_yaml), f"Missing {dec_yaml}"

    ac_cfg  = _load_yaml(ac_yaml)    # {app,data,model,mask,meta,...}
    dec_cfg = _load_yaml(dec_yaml)

    # Resolve ckpt files
    # Priority: args_eval.model.*  > YAML meta.pre_train_model > auto-find under folder
    mdl_in = args_eval.get("model", {}) if isinstance(args_eval.get("model", {}), dict) else {}

    ac_pth = mdl_in.get("ac_jepa_pth") or ac_cfg.get("meta", {}).get("pre_train_model")
    ac_pth = _join_if_relative(ac_dir, ac_pth)
    if not ac_pth or not os.path.exists(ac_pth):
        ac_pth = _find_ckpt_file(ac_dir)

    dec_pth = mdl_in.get("decoder_pth") or dec_cfg.get("meta", {}).get("pre_train_model")
    dec_pth = _join_if_relative(dec_dir, dec_pth)
    if not dec_pth or not os.path.exists(dec_pth):
        dec_pth = _find_ckpt_file(dec_dir)

    if not ac_pth:
        logger.warning(f"No AC-JEPA checkpoint found under {ac_dir}")
    if not dec_pth:
        logger.warning(f"No decoder checkpoint found under {dec_dir}")

    # Merge sections back to args_eval for the rest of your code:
    # Data: prefer AC for core geometry (num_frames/patch/tubelet/crop); decoder for eval_datasets
    ac_data  = ac_cfg.get("data", {})
    dec_data = dec_cfg.get("data", {})
    user_data = args_eval.get("data", {})
    merged_data = {**dec_data, **ac_data, **user_data}
    if user_data.get("eval_datasets"):
        merged_data["eval_datasets"] = user_data["eval_datasets"]


    # Model: prefer AC for predictor/action; decoder for decoder hyperparams
    ac_model  = ac_cfg.get("model", {})
    dec_model = dec_cfg.get("model", {})
    merged_model = {**dec_model, **ac_model}
    merged_model["ac_jepa_pth"] = ac_pth
    merged_model["decoder_pth"] = dec_pth

    # Mask from AC (decoder通常沒有)
    merged_mask = ac_cfg.get("mask", args_eval.get("mask", []))

    # Meta: user eval.yaml overrides component YAMLs
    merged_meta = {**dec_cfg.get("meta", {}), **ac_cfg.get("meta", {}), **args_eval.get("meta", {})}

    # Write back
    args_eval["data"]  = merged_data
    args_eval["model"] = merged_model
    args_eval["mask"]  = merged_mask
    args_eval["meta"]  = merged_meta

    # Also return raw yaml dicts if needed
    return args_eval, ac_cfg, dec_cfg

def _strip_module_prefix(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def _load_component(checkpoint, name, model, use_ddp):
    """
    Load a single submodule state dict from a composite checkpoint.

    Args:
        checkpoint (dict): A dictionary containing submodule state dicts
                           under keys like 'encoder', 'target_encoder', etc.
        name (str): The submodule name to load, e.g. 'encoder'.
        model (nn.Module): The instantiated model/submodule to load into.
        use_ddp (bool): If True, assume checkpoint keys are already aligned
                        with DDP ("module.") prefixes and DO NOT strip them.
                        If False, strip leading "module." from keys.

    Returns:
        nn.Module: The same model instance after attempting to load weights.
    """
    if model is not None and name in checkpoint:
        try:
            # NOTE:
            # - When training with DDP, checkpoints often store parameter
            #   names prefixed by "module.".
            # - If `use_ddp` is True, we keep keys as-is.
            # - If `use_ddp` is False, we strip the "module." prefix to match
            #   non-DDP module param names.
            ckpt = checkpoint[name] if use_ddp else _strip_module_prefix(checkpoint[name])
            msg = model.load_state_dict(ckpt)
            logger.info(f'Loaded {name} with msg: {msg}')
        except Exception as e:
            logger.warning(f'Failed to load {name}: {e}')
    else:
        logger.warning(f'No "{name}" found in checkpoint.')
    return model

def _set_trainability(module: nn.Module, trainable: bool, eval_when_frozen: bool = True):
    """
    Turn gradients on/off for a module in a unified way.
    - When trainable=False, set requires_grad(False) for all params and optionally call eval()
      to stop BatchNorm running stats from updating.
    - When trainable=True, set requires_grad(True) for all params and call train().
    """
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad_(trainable)
    if trainable:
        module.train()
    else:
        if eval_when_frozen:
            module.eval()

def load_pretrained_model(
    model_path,
    encoder,
    target_encoder,
    predictor,
    latent_action_enc,
    decoder,
    trainable: bool = True,
    use_ddp: bool = True
):
    """
    Load pretrained weights for all modules and apply a unified gradient on/off switch.

    Args:
        model_path (dict): Paths to checkpoints.
            Required keys:
                - 'ac_jepa_pth': str
                    Checkpoint containing keys:
                        'encoder', 'target_encoder', 'predictor', 'latent_action_encoder'
                - 'decoder_pth': str
                    Either:
                        a) checkpoint with a 'decoder' key, or
                        b) a direct state_dict compatible with decoder.load_state_dict.
        encoder, target_encoder, predictor, latent_action_enc, decoder (nn.Module): Modules to load.
        trainable (bool): Unified switch. If False, all modules will have requires_grad(False)
                          and be set to eval() to freeze BatchNorm running stats.
        use_ddp (bool): Keep 'module.' prefixes if True; otherwise strip them.

    Returns:
        tuple: (encoder, target_encoder, predictor, latent_action_enc, decoder)
    """

    # ---- 1) Load AC-JEPA-side modules ----
    ac_ckpt, ac_path = None, None
    try:
        ac_path = model_path.get('ac_jepa_pth', None) if isinstance(model_path, dict) else None
        if ac_path:
            ac_ckpt = torch.load(ac_path, map_location=torch.device('cpu'))
        else:
            logger.warning("No 'ac_jepa_pth' provided in model_path; skipping AC-JEPA modules.")
    except Exception as e:
        logger.info(f'Exception when loading AC-JEPA checkpoint from "{ac_path}": {e}')

    if ac_ckpt is not None:
        try:
            for name, module in [
                ('encoder', encoder),
                ('target_encoder', target_encoder),
                ('predictor', predictor),
                ('latent_action_encoder', latent_action_enc),
            ]:
                _ = _load_component(ac_ckpt, name, module, use_ddp)
        except Exception as e:
            logger.info(f'Failed to load AC-JEPA modules from checkpoint: {e}')
    else:
        logger.warning('AC-JEPA checkpoint not loaded; encoder/target_encoder/predictor/latent_action_enc remain as-initialized.')

    # ---- 2) Load decoder module ----
    dec_ckpt, dec_path = None, None
    try:
        dec_path = model_path.get('decoder_pth', None) if isinstance(model_path, dict) else None
        if dec_path:
            dec_ckpt = torch.load(dec_path, map_location=torch.device('cpu'))
        else:
            logger.warning("No 'decoder_pth' provided in model_path; skipping decoder load.")
    except Exception as e:
        logger.info(f'Exception when loading decoder checkpoint from "{dec_path}": {e}')

    if dec_ckpt is not None:
        _ = _load_component(dec_ckpt, 'decoder', decoder, use_ddp)
    else:
        logger.warning('Decoder checkpoint not loaded; decoder remains as-initialized.')

    # ---- 3) Apply unified trainability to ALL modules ----
    for m in (encoder, target_encoder, predictor, latent_action_enc, decoder):
        _set_trainability(m, trainable=trainable, eval_when_frozen=True)

    return encoder, target_encoder, predictor, latent_action_enc, decoder


def init_models(
    device: torch.device,
    video_model_params: dict,
    la_enc_params: dict,
    decoder_params: dict,
):
    """
    Initialize encoder, predictor, latent action encoder, and decoder modules.

    Args:
        device (torch.device): Torch device to place the models on.
        video_model_params (dict): Parameters for initializing the video model.
            Required keys:
                - patch_size (int)
                - num_frames (int)
                - tubelet_size (int)
                - model_name (str)
                - uniform_power (bool)
                - use_sdpa (bool)
                - use_mask_tokens (bool)
                - num_mask_tokens (int)
                - zero_init_mask_tokens (bool)
                - crop_size (int)
                - pred_depth (int)
                - pred_embed_dim (int)
                - adapter_type (str)
                - action_dim (int)
        la_enc_params (dict): Parameters for initializing the latent action encoder.
            Required keys:
                - num_heads (int)
                - d_codebook (int)
                - n_codebook (int)
                - vq_bias (bool)
                - vq_commit_weight (float)
                - vq_entropy_weight (float)
                - vq_diversity_weight (float)
        decoder_params (dict): Parameters for initializing the video decoder.
            Required keys:
                - img_size (int)
                - patch_size (int)
                - num_frames (int)
                - tubelet_size (int)
                - in_channels (int)
                - depth (int)
                - num_heads (int)
                - mlp_ratio (float)
                - norm_layer (nn.Module)

    Returns:
        tuple: (encoder, predictor, latent_action_enc, decoder)
    """

    # --------------------------
    # 1. Video Encoder & Predictor
    # --------------------------
    encoder, predictor = init_video_model(
        uniform_power=video_model_params["uniform_power"],
        use_mask_tokens=video_model_params["use_mask_tokens"],
        num_mask_tokens=video_model_params["num_mask_tokens"],
        zero_init_mask_tokens=video_model_params["zero_init_mask_tokens"],
        device=device,
        patch_size=video_model_params["patch_size"],
        num_frames=video_model_params["num_frames"],
        tubelet_size=video_model_params["tubelet_size"],
        model_name=video_model_params["model_name"],
        crop_size=video_model_params["crop_size"],
        pred_depth=video_model_params["pred_depth"],
        pred_embed_dim=video_model_params["pred_embed_dim"],
        use_sdpa=video_model_params["use_sdpa"],
        adapter_type=video_model_params["adapter_type"],
    )
    target_encoder = copy.deepcopy(encoder)

    # --------------------------
    # 2. Latent Action Encoder
    # --------------------------
    latent_action_enc = init_latent_action_encoder(
        device=device,
        inp_dims=encoder.backbone.embed_dim, 
        num_heads=la_enc_params["num_heads"],
        d_codebook=la_enc_params["d_codebook"],
        n_codebook=la_enc_params["n_codebook"],
        vq_bias=la_enc_params["vq_bias"],
        vq_commit_weight=la_enc_params["vq_commit_weight"],
        vq_entropy_weight=la_enc_params["vq_entropy_weight"],
        vq_diversity_weight=la_enc_params["vq_diversity_weight"],
    )

    # --------------------------
    # 3. Decoder
    # --------------------------
    decoder = ViTVideoDecoder(
        img_size=decoder_params["img_size"],
        patch_size=decoder_params["patch_size"],
        num_frames=decoder_params["num_frames"],
        tubelet_size=decoder_params["tubelet_size"],
        in_channels=decoder_params["in_channels"],
        in_dim=encoder.backbone.embed_dim,
        embed_dim=encoder.backbone.embed_dim // 2,
        depth=decoder_params["depth"],
        num_heads=decoder_params["num_heads"],
        mlp_ratio=decoder_params["mlp_ratio"],
        norm_layer=decoder_params["norm_layer"],
    )
    decoder.to(device)
    logger.info(decoder)

    return encoder, target_encoder, predictor, latent_action_enc, decoder


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
)->tuple[MultiMaskWrapper, PredictorMultiMaskWrapper]:
    
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

    encoder.to(device)
    predictor.to(device)

    logger.info(encoder)
    logger.info(predictor)

    return encoder, predictor

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
    )-> LatentActionEncoderMultiMaskWrapper:
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
    
    la_enc = la_enc.to(device)
    logger.info(la_enc)

    return la_enc

# This part are same vit_decoder/utils.py
def unpatchify(
    patches: torch.Tensor,
    img_size: int,
    num_frames: int,
    patch_size: int,
    tubelet_size: int,
) -> torch.Tensor:
    """
    Unpatchifies video clips from non-overlapping **flattened** 3D patches.

    Args:
        patches: A tensor of shape (B, L, T * P * P * C), where each element is a flattened patch.
               - B: batch size
               - L: the total number of patches (L = (F/T) * (H/P) * (W/P))
               - T: tubelet_size
               - P: patch_size
               - C: number of channels
        img_size: The height and width of the original image.
        num_frames: The number of frames in the original video.
        patch_size: The height and width of the 2D spatial patches.
        tubelet_size: The number of consecutive frames to include in each temporal patch (tubelet).

    Returns:
        A tensor of shape (B, C, F, H, W), where each element is a frame.
        - B: batch size
        - C: number of channels
        - F: number of frames
        - H: image height
        - W: image width
    """
    clips = rearrange(
        patches,
        "b (f h w) (t p1 p2 c) -> b c (f t) (h p1) (w p2)",
        f=num_frames // tubelet_size,
        h=img_size // patch_size,
        w=img_size // patch_size,
        t=tubelet_size,
        p1=patch_size,
        p2=patch_size,
    )
    return clips
