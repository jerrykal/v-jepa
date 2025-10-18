
import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass


import pprint
import logging
import numpy as np
import imageio.v2 as imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import random

from einops import rearrange
from torch.nn.parallel import DistributedDataParallel
from src.utils.distributed import AllReduce, init_distributed
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.utils import apply_masks

from src.utils.tensors import unnormalize_tensor
from src.datasets.data_manager import init_data
from src.utils.logging import (
    get_logger)
from evals.ac_visualization_frozen.utils import(
    init_models,
    load_pretrained_model,
    merge_eval_args_with_components,
    unpatchify
)

from app.vit_decoder.transforms import (
    make_eval_transforms,
)

pp = pprint.PrettyPrinter(indent=4)

_GLOBAL_SEED = 123
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__)

def build_masks(B:int, t:int, p:int, device):
    mask = torch.ones((t, p), dtype=torch.int32).to(device)
    mask[-1, :] = 0
    mask = mask.flatten()
    mask_p = torch.argwhere(mask == 0).reshape(1, -1).expand(B, -1)
    mask_e = torch.nonzero(mask).reshape(1, -1).expand(B, -1)
    return [mask_e], [mask_p]

def _extract_clips_from_batch(batch):
    """
    Return a 5D tensor (B,C,T,H,W) from a dataloader batch.
    Tries common keys, else assume the batch itself is a tensor.
    """
    if isinstance(batch, dict):
        for k in ['video', 'videos', 'clip', 'clips', 'frames', 'imgs', 'images', 'pixel_values']:
            if k in batch:
                x = batch[k]
                break
        else:
            x = next(iter(batch.values()))
    elif isinstance(batch, (list, tuple)):
        x = batch[0]
    else:
        x = batch

    assert torch.is_tensor(x), f"Batch does not contain a tensor, got type={type(x)}"
    assert x.dim() == 5, f"Expected a 5D video tensor, got {tuple(x.shape)}"
    
    # expect B,C,T,H,W ; if B,T,C,H,W -> permute
    if x.shape[1] in (1, 3):
        return x
    elif x.shape[2] in (1, 3):
        return x.permute(0, 2, 1, 3, 4).contiguous()
    else:
        return x
    
def _to_uint8_video(x):
    # x: (T,H,W,C) float -> uint8
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).round().astype(np.uint8)
    return x

def _write_gif(save_dir, stem, video_thwc_uint8, fps=4, loop=0):
    os.makedirs(save_dir, exist_ok=True)
    gif_path = os.path.join(save_dir, f"{stem}.gif")
    duration = 1.0 / max(1, fps)
    imageio.mimsave(gif_path, list(video_thwc_uint8), format="GIF", duration=duration, loop=loop)
    return gif_path

def _save_last_k_frames(save_dir, stem, video_uint8, k=2):
    T = video_uint8.shape[0]
    k = max(1, min(k, T))
    for t in range(T - k, T):
        out_path = os.path.join(save_dir, f"{stem}_lastf_{t:04d}.png")
        imageio.imwrite(out_path, video_uint8[t])
    return k

def _side_by_side(a, b):
    T0 = min(a.shape[0], b.shape[0])
    H0 = min(a.shape[1], b.shape[1])
    W0 = min(a.shape[2], b.shape[2])
    return np.concatenate([a[:T0, :H0, :W0], b[:T0, :H0, :W0]], axis=2)

def main(args_eval, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    args_eval, ac_raw, dec_raw = merge_eval_args_with_components(args_eval)
    
    cfgs_meta  = args_eval.get('meta', {})
    cfgs_data  = args_eval.get('data', {})
    cfgs_model = args_eval.get('model', {})
    cfgs_mask  = args_eval.get('mask', [])

    num_frames   = cfgs_data.get('num_frames')
    tubelet_size = cfgs_data.get('tubelet_size')
    patch_size   = cfgs_data.get('patch_size')
    crop_size    = cfgs_data.get('crop_size', 224)

    # model pieces
    video_model_params = {
        "uniform_power":       cfgs_model.get('uniform_power', True),
        "use_mask_tokens":     cfgs_model.get('use_mask_tokens', True),
        "num_mask_tokens":     len(cfgs_mask) if isinstance(cfgs_mask, (list, tuple)) else 1,
        "zero_init_mask_tokens": cfgs_model.get('zero_init_mask_tokens', True),
        "patch_size":          patch_size,
        "num_frames":          num_frames,
        "tubelet_size":        tubelet_size,
        "model_name":          cfgs_model.get('model_name'),
        "crop_size":           crop_size,
        "pred_depth":          cfgs_model.get('pred_depth'),
        "pred_embed_dim":      cfgs_model.get('pred_embed_dim'),
        "use_sdpa":            cfgs_meta.get('use_sdpa', False),
        "adapter_type":        cfgs_model.get('action_adapter_type', "None"),
    }

    la_enc_params = {
        "num_heads":           cfgs_model.get('latent_action_num_heads', 8),
        "d_codebook":          cfgs_model.get('dims_aciton_codebook', 32),
        "n_codebook":          cfgs_model.get('number_aciton_codebook', 1),
        "vq_bias":             cfgs_model.get('vq_bias', True),
        "vq_commit_weight":    cfgs_model.get('vq_commit_weight', 0.25),
        "vq_entropy_weight":   cfgs_model.get('vq_entropy_weight', 0.1),
        "vq_diversity_weight": cfgs_model.get('vq_diversity_weight', 1.0),
    }

    decoder_params = {
        "img_size":   crop_size,
        "patch_size": patch_size,
        "num_frames": num_frames,
        "tubelet_size": tubelet_size,
        "in_channels": 3,
        "depth":      cfgs_model.get('decoder_depth', 8),
        "num_heads":  cfgs_model.get('decoder_num_heads', 16),
        "mlp_ratio":  cfgs_model.get('decoder_mlp_ratio', 4.0),
        "norm_layer": nn.LayerNorm,
    }

    dataset_type         = cfgs_data.get('dataset_type', 'VideoDataset')
    eval_dataset_paths   = cfgs_data.get('eval_datasets', [])
    batch_size           = cfgs_data.get('batch_size', 1)
    num_clips            = cfgs_data.get('num_clips', 1)
    num_frames           = cfgs_data.get('num_frames')
    tubelet_size         = cfgs_data.get('tubelet_size')
    sampling_rate        = cfgs_data.get('sampling_rate')
    duration             = cfgs_data.get('clip_duration', None)
    crop_size            = cfgs_data.get('crop_size', 224)
    patch_size           = cfgs_data.get('patch_size')
    pin_mem              = cfgs_data.get('pin_mem', False)
    num_workers          = cfgs_data.get('num_workers', 1)
    filter_short_videos  = cfgs_data.get('filter_short_videos', False)
    decode_one_clip      = cfgs_data.get('decode_one_clip', True)
    log_resource_util_data = cfgs_data.get('log_resource_utilization', False)

    folder              = args_eval.get('logging', {}).get('folder', None)
    enable_autoregressive = cfgs_meta.get('enable_autoregressive', False)
    ac_jepa_pth = cfgs_model.get("ac_jepa_pth")
    decoder_pth = cfgs_model.get("decoder_pth")
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    # dtype / mixed precision
    which_dtype = cfgs_meta.get('dtype', 'float32')
    logger.info(f'{which_dtype=}')
    if str(which_dtype).lower() == 'bfloat16':
        dtype = torch.bfloat16
        mixed_precision = True
    elif str(which_dtype).lower() in ('float16', 'fp16', 'half'):
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- init model
    encoder, target_encoder, predictor, latent_action_enc, decoder = init_models(
        device=device,
        video_model_params=video_model_params,
        la_enc_params=la_enc_params,
        decoder_params=decoder_params,
    )
    model_dict = {
        "encoder":encoder, 
        "target_encoder":target_encoder, 
        "predictor":predictor, 
        "latent_action_enc":latent_action_enc, 
        "decoder":decoder
        }

    # -- make data transforms
    eval_transform = make_eval_transforms(crop_size=crop_size)

    # -- init data-loaders/samplers
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

    # -- freeze encoder
    for k, m in model_dict.items():
        for p in m.parameters():
            p.requires_grad = False
        print(f"Freeze the {k}")

    # -- load pretrainig model checkpoint
    encoder, target_encoder, predictor, latent_action_enc, decoder \
    = load_pretrained_model(
        model_path={
            'ac_jepa_pth':ac_jepa_pth,
            'decoder_pth':decoder_pth
        },
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        latent_action_enc=latent_action_enc,
        decoder=decoder,
        trainable=False,
        use_ddp=False,
    )

    # -- wrap decoder with DDP(optional)


    # -- Evaluation 
    # >> 1. sample from dataset
    # >> 2. ac model predict the next frame
    #   a. x = encoder(clips)
    #   b. p = predictor(x,mask)
    # >> 3. feature_recon = decoder(x)
    # >> 4. predicted_recon = decoder(p)
    # >> 5. save video
    #   a. save clips
    #   b. save feature_recon
    #   c. save predicted_recon

    results_dir = os.path.join(folder if folder is not None else ".", "eval_outputs")
    os.makedirs(results_dir, exist_ok=True)

    encoder.eval()
    predictor.eval()
    decoder.eval()
    latent_action_enc.eval()

    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
        eval_loader_iter = iter(eval_loader)
        first_batch = next(eval_loader_iter)[0]
        clips = _extract_clips_from_batch(first_batch).to(device=device)
        B, C, T, H, W = clips.shape
        _t = T // tubelet_size
        _p = (crop_size // patch_size) ** 2 
        masks_e, masks_p = build_masks(B=B, t=_t, p=_p, device=clips.device)
        
        def forward_target(c):
            """
            Returns list of tensors of shape [B, N, D], one for each
            mask-pred.
            """
            with torch.no_grad():
                h = target_encoder(c)
                h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]

                # -- create targets (masked regions of h)
                # masked_h = apply_masks(h, masks_p, concat=False)
                return [h]*len(masks_p)

            
        def foward_latent_action_encode(h):
            with torch.no_grad():
                x = [rearrange(_h, "B (t p) D -> B t p D", t=num_frames//tubelet_size, p=(crop_size//patch_size)**2) for _h in h]
                results = latent_action_enc(x)
                act_list, _ = zip(*results)  
                return list(act_list)
            
        def forward_context(c, act, autoregressive: bool = False,):
            """
            Returns list of tensors of shape [B, N, D], one for each
            mask-pred.
            """
            with torch.no_grad():
                z_list = encoder(c, masks_e)
                z_ctx = z_list[0]
                if autoregressive:
                    preds_tokens = [] 
                    preds_tokens.append(z_ctx[:, :_p])
                    for _ in range(_t-1):
                        out_list = predictor([z_ctx], None, masks_e, masks_p, act)
                        pred_tok = out_list[0]
                        preds_tokens.append(pred_tok)
                        z_ctx = torch.cat([z_ctx[:, _p:], pred_tok], dim=1) 
                    full_z = torch.cat(preds_tokens, dim=1)

                else:
                    out_list = predictor([z_ctx], None, masks_e, masks_p, act)
                    pred_tok = out_list[0]
                    full_z = torch.concat([z_ctx, pred_tok],dim=1)
                return full_z
        
        target_feature      = forward_target(clips)
        action              = foward_latent_action_encode(target_feature)
        full_z              = forward_context(clips, action, autoregressive=enable_autoregressive)
        # features            = F.layer_norm(full_z, (full_z. size(-1),))

        preds               = decoder(full_z) 
        preds               = unpatchify(preds, crop_size, num_frames, patch_size, tubelet_size)
        preds               = unnormalize_tensor(preds) 
        preds               = preds.clamp(0.0, 1.0)
        preds               = rearrange(preds, "b c f h w -> (b f) h w c")
        pred_np             = preds.detach().to(torch.float32).cpu().numpy()

        targets             = unnormalize_tensor(clips)
        targets             = rearrange(targets, "b c f h w -> (b f) h w c")
        target_np           = targets.detach().to(torch.float32).cpu().numpy()

    # >> Save image to results_dir
    stem_base = f"sample"
    pred_np   = pred_np.reshape(B, T, H, W, C)
    target_np = target_np.reshape(B, T, H, W, C)

    # Fetch first clips
    vid_pred   = _to_uint8_video(pred_np[0])
    vid_target = _to_uint8_video(target_np[0]) 
    side       = _side_by_side(vid_target, vid_pred)

    # Save gif 
    fps = 4
    last_k = min(tubelet_size, T)
    out_gt_gif    = _write_gif(results_dir, f"{stem_base}_gt",   vid_target, fps=fps, loop=0)
    out_pred_gif  = _write_gif(results_dir, f"{stem_base}_pred", vid_pred,   fps=fps, loop=0)
    out_side_gif  = _write_gif(results_dir, f"{stem_base}_side", side,       fps=fps, loop=0)

    saved_k_gt   = _save_last_k_frames(results_dir, f"{stem_base}_gt",   vid_target, k=last_k)
    saved_k_pred = _save_last_k_frames(results_dir, f"{stem_base}_pred", vid_pred,   k=last_k)

    logger.info(f"[Save] GIF GT   -> {out_gt_gif}")
    logger.info(f"[Save] GIF PRED -> {out_pred_gif}")
    logger.info(f"[Save] GIF SIDE -> {out_side_gif}")
    logger.info(f"[Save] last {saved_k_gt} GT frames & last {saved_k_pred} PRED frames as PNGs")