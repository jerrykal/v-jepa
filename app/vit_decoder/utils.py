# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import src.models.vision_transformer as video_vit
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from einops import rearrange
from src.models.utils.multimask import MultiMaskWrapper
from src.models.vit_decoder import ViTVideoDecoder
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_jepa_encoder(model_path: str, encoder: nn.Module) -> None:
    """
    Load JEPA encoder weights in place.

    Args:
        model_path: str, path to the JEPA encoder checkpoint
        encoder: nn.Module, the encoder to load the weights into
    """
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
                pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}

            msg = encoder.load_state_dict(pretrained_dict)
            logger.info(f"Loaded JEPA encoder with msg: {msg}")
        else:
            logger.warning('No "encoder" found in checkpoint.')

    except Exception as e:
        logger.info(f"Failed to load JEPA encoder: {e}")


def load_checkpoint(
    r_path: str,
    decoder: ViTVideoDecoder,
    opt: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> int:
    """
    Load decoder weights as well as optimizer and scaler states in place.

    Args:
        r_path: str, path to the decoder checkpoint
        decoder: ViTVideoDecoder, the decoder to load the weights into
        opt: torch.optim.Optimizer, the optimizer to load the weights into
        scaler: torch.cuda.amp.GradScaler, the scaler to load the weights into

    Returns:
        int, the epoch of the checkpoint
    """
    try:
        checkpoint = torch.load(r_path, map_location=torch.device("cpu"), weights_only=False)
    except Exception as e:
        logger.info(f"Encountered exception when loading checkpoint {e}")

    epoch = 0
    try:
        epoch = checkpoint["epoch"]

        # NOTE: since the pre-trained weights are saved with DDP wrapper, we need to remove the DDP prefix "module." if we are not using DDP
        if not isinstance(decoder, DistributedDataParallel):
            checkpoint["decoder"] = {k.replace("module.", ""): v for k, v in checkpoint["decoder"].items()}

        # -- loading decoder
        pretrained_dict = checkpoint["decoder"]
        msg = decoder.load_state_dict(pretrained_dict)
        logger.info(f"loaded pretrained decoder from epoch {epoch} with msg: {msg}")

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
    model_name: str = "vit_huge",
    img_size: int = 224,
    uniform_power: bool = False,
    use_sdpa: bool = False,
    in_channels: int = 3,
    decoder_depth: int = 8,
    decoder_num_heads: int = 16,
    decoder_mlp_ratio: float = 4.0,
    decoder_norm_layer: nn.Module = nn.LayerNorm,
    encode_frames_independently: bool = False,
) -> tuple[MultiMaskWrapper, ViTVideoDecoder]:
    """
    Initialize encoder and decoder models.

    Args:
        device: torch.device, the device to run the models on
        patch_size: int, the size of the patches to use for the encoder
        num_frames: int, the number of frames to use for the encoder
        tubelet_size: int, the size of the tubelets to use for the encoder
        model_name: str, the name of the model to use for the encoder
        img_size: int, the size of the images to use for the encoder
        uniform_power: bool, whether to use uniform power for the encoder
        use_sdpa: bool, whether to use SDPA for the encoder
        in_channels: int, the number of channels to use for the decoder
        decoder_depth: int, the depth of the decoder
        decoder_num_heads: int, the number of heads to use for the decoder
        decoder_mlp_ratio: float, the ratio of the decoder MLP
        decoder_norm_layer: nn.Module, the normalization layer to use for the decoder

    Returns:
        tuple[MultiMaskWrapper, ViTVideoDecoder], the encoder and decoder models
    """
    encoder = video_vit.__dict__[model_name](
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
    )
    encoder = MultiMaskWrapper(encoder)

    decoder = ViTVideoDecoder(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size if not encode_frames_independently else 1,
        in_channels=in_channels,
        in_dim=encoder.backbone.embed_dim,
        embed_dim=encoder.backbone.embed_dim // 2,
        depth=decoder_depth,
        num_heads=decoder_num_heads,
        mlp_ratio=decoder_mlp_ratio,
        norm_layer=decoder_norm_layer,
    )

    encoder.to(device)
    decoder.to(device)
    logger.info(encoder)
    logger.info(decoder)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Decoder number of parameters: {count_parameters(decoder)}")

    return encoder, decoder


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
    """
    Initialize optimizer and scaler.

    Args:
        models: list[nn.Module], the models to optimize
        scheduler_type: str, the type of scheduler to use
        iterations_per_epoch: int, the number of iterations per epoch
        lr: float, the learning rate
        warmup: float, the warmup ratio
        num_epochs: int, the number of epochs
        wd: float, the weight decay
        mixed_precision: bool, whether to use mixed precision
        betas: tuple[float, float], the betas for the optimizer
        eps: float, the epsilon for the optimizer
        zero_init_bias_wd: bool, whether to exclude bias parameters from weight decay

    Returns:
        tuple[torch.optim.AdamW, torch.cuda.amp.GradScaler | None, LambdaLR], the optimizer, scaler, and scheduler
    """
    param_groups = []
    for model in models:
        param_groups.append(
            {"params": (p for n, p in model.named_parameters() if ("bias" not in n) and (len(p.shape) != 1))}
        )
        param_groups.append(
            {
                "params": (p for n, p in model.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
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


def patchify(clips: torch.Tensor, patch_size: int, tubelet_size: int) -> torch.Tensor:
    """
    Patchifies video clips into non-overlapping **flattened** 3D patches.

    Args:
        clips: A 5D tensor of shape (B, C, F, H, W) representing a batch of videos.
               - B: batch size
               - C: number of channels
               - F: number of frames
               - H: image height
               - W: image width
        patch_size: The height and width of the 2D spatial patches.
        tubelet_size: The number of consecutive frames to include in each temporal patch (tubelet).

    Returns:
        A tensor of shape (B, L, T * P * P * C), where each element is a flattened patch.
        - B: batch size
        - L: the total number of patches (L = (F/T) * (H/P) * (W/P))
        - T: tubelet_size
        - P: patch_size
        - C: number of channels
    """
    patches = rearrange(
        clips,
        "b c (f t) (h p1) (w p2) -> b (f h w) (t p1 p2 c)",
        t=tubelet_size,
        p1=patch_size,
        p2=patch_size,
    )
    return patches


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
