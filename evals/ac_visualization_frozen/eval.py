import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass


import pprint

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.vit_decoder.transforms import (
    make_eval_transforms,
)
from einops import rearrange
from evals.ac_visualization_frozen.utils import (
    init_models,
    load_pretrained_model,
    merge_eval_args_with_components,
    unpatchify,
)
from src.datasets.data_manager import init_data
from src.utils.distributed import init_distributed
from src.utils.logging import get_logger
from src.utils.tensors import unnormalize_tensor
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=4)

_GLOBAL_SEED = 123
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__)


def build_masks(B: int, t: int, p: int, device):
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
        for k in ["video", "videos", "clip", "clips", "frames", "imgs", "images", "pixel_values"]:
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

    cfgs_meta = args_eval.get("meta", {})
    cfgs_data = args_eval.get("data", {})
    cfgs_model = args_eval.get("model", {})

    num_frames = cfgs_data.get("num_frames")
    # Double number of frames, the first half serves as context and the second half serves as target
    clip_len = num_frames * 2
    tubelet_size = cfgs_data.get("tubelet_size")
    patch_size = cfgs_data.get("patch_size")
    crop_size = cfgs_data.get("crop_size", 224)
    num_patches_per_frame = (crop_size // patch_size) ** 2

    num_samples_to_generate = cfgs_meta.get("num_samples_to_generate", 1)

    # model pieces
    video_model_params = {
        "uniform_power": cfgs_model.get("uniform_power", True),
        "patch_size": patch_size,
        "num_frames": num_frames,
        "tubelet_size": tubelet_size,
        "model_name": cfgs_model.get("model_name"),
        "crop_size": crop_size,
        "pred_depth": cfgs_model.get("pred_depth"),
        "pred_embed_dim": cfgs_model.get("pred_embed_dim"),
        "use_sdpa": cfgs_meta.get("use_sdpa", False),
    }

    la_enc_params = {
        "num_heads": cfgs_model.get("latent_action_num_heads", 8),
        "d_codebook": cfgs_model.get("dims_aciton_codebook", 32),
        "n_codebook": cfgs_model.get("number_aciton_codebook", 1),
        "vq_bias": cfgs_model.get("vq_bias", True),
        "vq_commit_weight": cfgs_model.get("vq_commit_weight", 0.25),
        "vq_entropy_weight": cfgs_model.get("vq_entropy_weight", 0.1),
        "vq_diversity_weight": cfgs_model.get("vq_diversity_weight", 1.0),
        "use_sdpa": cfgs_meta.get("use_sdpa", False),
        "num_patches_per_frame": num_patches_per_frame,
    }

    decoder_params = {
        "img_size": crop_size,
        "patch_size": patch_size,
        "num_frames": num_frames,
        "tubelet_size": tubelet_size,
        "in_channels": 3,
        "depth": cfgs_model.get("decoder_depth", 8),
        "num_heads": cfgs_model.get("decoder_num_heads", 16),
        "mlp_ratio": cfgs_model.get("decoder_mlp_ratio", 4.0),
        "norm_layer": nn.LayerNorm,
    }

    dataset_type = cfgs_data.get("dataset_type", "VideoDataset")
    eval_dataset_paths = cfgs_data.get("eval_datasets", [])
    batch_size = cfgs_data.get("batch_size", 1)
    num_clips = cfgs_data.get("num_clips", 1)
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

    folder = args_eval.get("logging", {}).get("folder", None)
    autoregressive_steps = cfgs_meta.get("autoregressive_steps", 0)
    ac_jepa_pth = cfgs_model.get("ac_jepa_pth")
    decoder_pth = cfgs_model.get("decoder_pth")
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    # dtype / mixed precision
    which_dtype = cfgs_meta.get("dtype", "float32")
    logger.info(f"{which_dtype=}")
    if str(which_dtype).lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif str(which_dtype).lower() in ("float16", "fp16", "half"):
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
    encoder, target_encoder, ac_predictor, latent_action_enc, decoder = init_models(
        device=device,
        video_model_params=video_model_params,
        la_enc_params=la_enc_params,
        decoder_params=decoder_params,
    )
    model_dict = {
        "encoder": encoder,
        "target_encoder": target_encoder,
        "ac_predictor": ac_predictor,
        "latent_action_enc": latent_action_enc,
        "decoder": decoder,
    }

    # -- make data transforms
    eval_transform = make_eval_transforms(crop_size=crop_size)

    # -- init data-loaders/samplers
    eval_loader, _ = init_data(
        data=dataset_type,
        root_path=eval_dataset_paths,
        batch_size=batch_size,
        training=False,
        clip_len=clip_len,
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
    encoder, target_encoder, ac_predictor, latent_action_enc, decoder = load_pretrained_model(
        model_path={"ac_jepa_pth": ac_jepa_pth, "decoder_pth": decoder_pth},
        encoder=encoder,
        target_encoder=target_encoder,
        ac_predictor=ac_predictor,
        latent_action_enc=latent_action_enc,
        decoder=decoder,
        trainable=False,
        use_ddp=False,
    )

    # -- wrap decoder with DDP(optional)

    # -- Evaluation
    results_dir = os.path.join(folder if folder is not None else ".", "eval_outputs")
    os.makedirs(results_dir, exist_ok=True)

    encoder.eval()
    ac_predictor.eval()
    decoder.eval()
    latent_action_enc.eval()

    eval_loader = iter(eval_loader)

    for i in tqdm(range(num_samples_to_generate), desc="Generating samples"):
        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
            first_batch = next(eval_loader)[0]
            clips = _extract_clips_from_batch(first_batch).to(device=device)
            B, C, T, H, W = clips.size()
            T //= 2

            # Rearrange clips from (B, C, T, H, W) to (B * T, C, 2, H, W),
            # flattening the temporal dimension into the batch so each frame can
            # be processed independently as an image. This enables the JEPA encoder
            # to operate over frames individually.
            context_x = rearrange(clips[:, :, :T, :, :], "b c t h w -> (b t) c 1 h w").repeat(1, 1, 2, 1, 1)
            target_x = rearrange(clips[:, :, T:, :, :], "b c t h w -> (b t) c 1 h w").repeat(1, 1, 2, 1, 1)

            def forward_encoder(clips):
                # Encode clips, then merge temporal dimension back with patch dimension
                features = encoder(clips)
                features = rearrange(features, "(b t) p d -> b t p d", t=num_frames)
                features = F.layer_norm(features, (features.size(-1),))  # normalize over feature-dim  [B, N, D]

                return features

            def forward_actions(targets):
                with torch.no_grad():
                    acts, _ = latent_action_enc(targets)
                    return acts

            def forward_predictions(contexts, acts_full):
                autoregressive_steps = acts_full.size(1) - contexts.size(1)

                all_preds: list = []
                preds = ac_predictor(contexts, acts_full[:, : contexts.size(1)])

                for step in range(autoregressive_steps):
                    next_frame_pred = preds[:, -num_patches_per_frame:]
                    all_preds.append(next_frame_pred)

                    # Shift preds
                    contexts = torch.cat([contexts, next_frame_pred.unsqueeze(1)], dim=1)
                    contexts = contexts[:, 1:]

                    acts_next = acts_full[:, step : step + contexts.size(1)]
                    preds = ac_predictor(contexts, acts_next)

                all_preds = torch.cat(all_preds, dim=1)
                return all_preds

            contexts = forward_encoder(context_x)
            targets = forward_encoder(target_x)
            acts_context = forward_actions(contexts)
            acts_target = forward_actions(targets)
            acts_full = torch.cat([acts_context, acts_target], dim=1)
            preds = forward_predictions(contexts, acts_full)

            preds = decoder(preds)
            preds = unpatchify(preds, crop_size, num_frames, patch_size, 1)
            preds = unnormalize_tensor(preds)
            preds = preds.clamp(0.0, 1.0)
            preds = rearrange(preds, "b c f h w -> (b f) h w c")
            preds_np = preds.detach().to(torch.float32).cpu().numpy()

            contexts = rearrange(contexts, "b t p d -> b (t p) d")
            contexts = decoder(contexts)
            contexts = unpatchify(contexts, crop_size, num_frames, patch_size, 1)
            contexts = unnormalize_tensor(contexts)
            contexts = contexts.clamp(0.0, 1.0)
            contexts = rearrange(contexts, "b c f h w -> (b f) h w c")
            contexts_np = contexts.detach().to(torch.float32).cpu().numpy()

            targets = rearrange(targets, "b t p d -> b (t p) d")
            targets = decoder(targets)
            targets = unpatchify(targets, crop_size, num_frames, patch_size, 1)
            targets = unnormalize_tensor(targets)
            targets = targets.clamp(0.0, 1.0)
            targets = rearrange(targets, "b c f h w -> (b f) h w c")
            targets_np = targets.detach().to(torch.float32).cpu().numpy()

        # >> Save image to results_dir
        stem_base = f"sample_{i}"
        preds_np = preds_np.reshape(B, T, H, W, C)
        contexts_np = contexts_np.reshape(B, T, H, W, C)
        targets_np = targets_np.reshape(B, T, H, W, C)

        # Fetch first clips
        vid_pred = _to_uint8_video(preds_np[0])
        vid_contexts = _to_uint8_video(contexts_np[0])
        vid_targets = _to_uint8_video(targets_np[0])
        vid_compare = _side_by_side(vid_pred, vid_targets)

        # Save gif
        fps = 4
        _write_gif(os.path.join(results_dir, "pred"), f"{stem_base}", vid_pred, fps=fps, loop=0)
        _write_gif(os.path.join(results_dir, "contexts"), f"{stem_base}", vid_contexts, fps=fps, loop=0)
        _write_gif(os.path.join(results_dir, "targets"), f"{stem_base}", vid_targets, fps=fps, loop=0)
        _write_gif(os.path.join(results_dir, "compare"), f"{stem_base}", vid_compare, fps=fps, loop=0)
