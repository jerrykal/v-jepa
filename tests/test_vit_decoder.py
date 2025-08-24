import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from app.vit_decoder.utils import (
    init_models,
    load_checkpoint,
    load_jepa_encoder,
    unpatchify,
)
from src.datasets.data_manager import init_data
from src.utils.logging import get_logger

logger = get_logger(__name__)


def unnormalize_tensor(
    tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        tensor: torch.Tensor, shape [B, C, F, H, W], values in [0, 1]
        mean: torch.Tensor, shape [C], mean values
        std: torch.Tensor, shape [C], std values

    Returns:
        torch.Tensor, shape [B, C, F, H, W], values in [0, 1]
    """
    tensor = tensor.clone().contiguous()
    mean = mean.to(tensor.device).unsqueeze(0)
    std = std.to(tensor.device).unsqueeze(0)

    B, C, F, H, W = tensor.shape
    tensor = tensor.view(B, C, -1).permute(0, 2, 1)
    tensor.mul_(std).add_(mean)
    tensor = tensor.permute(0, 2, 1).view(B, C, F, H, W)
    return tensor


def save_tensor_as_gif(
    tensor: torch.Tensor, output_path: str, duration: int = 100
) -> None:
    """
    Save a tensor of images as a GIF file.

    Args:
        tensor: torch.Tensor, shape [B, C, T, H, W], values in [0, 1]
        output_path: str, output GIF file path
        duration: int, duration of each frame in milliseconds, default is 100ms
    """
    # Convert from tensor to numpy and scale to 0-255
    tensor_np = tensor.cpu().numpy()
    tensor_np = (tensor_np * 255).astype(np.uint8)

    # Transpose from (C, T, H, W) to (T, H, W, C) for PIL
    tensor_np = tensor_np.transpose(1, 2, 3, 0)

    # Convert to PIL Images
    pil_images = [Image.fromarray(img) for img in tensor_np]

    # Save as gif
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0,
    )


@torch.no_grad()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    configs = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)

    # -- META
    cfgs_meta = configs.get("meta")
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", 42)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    which_dtype = cfgs_meta.get("dtype")
    pre_train_model = cfgs_meta.get("pre_train_model")
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
    cfgs_model = configs.get("model")
    model_name = cfgs_model.get("model_name")
    uniform_power = cfgs_model.get("uniform_power", True)
    decoder_depth = cfgs_model.get("decoder_depth", 8)
    decoder_num_heads = cfgs_model.get("decoder_num_heads", 16)
    decoder_mlp_ratio = cfgs_model.get("decoder_mlp_ratio", 4.0)

    # -- DATA
    cfgs_data = configs.get("data")
    dataset_type = cfgs_data.get("dataset_type", "videodataset")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights", None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), (
            "Must have one sampling weight specified for each dataset"
        )
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

    # -- LOGGING
    cfgs_logging = configs.get("logging")
    folder = cfgs_logging.get("folder")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    # Make data transforms
    short_side_size = int(crop_size * 256 / 224)
    normalize_mean = torch.tensor([0.485, 0.456, 0.406])
    normalize_std = torch.tensor([0.229, 0.224, 0.225])
    transform = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(crop_size, crop_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )

    # Init data-loaders/samplers
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=False,
        clip_len=num_frames,
        frame_sample_rate=sampling_rate,
        filter_short_videos=filter_short_videos,
        decode_one_clip=decode_one_clip,
        duration=duration,
        num_clips=num_clips,
        random_clip_sampling=False,
        transform=transform,
        datasets_weights=datasets_weights,
        collator=None,
        num_workers=num_workers,
        pin_mem=pin_mem,
    )
    logger.info("Initialized data-loaders/samplers")

    # Init model
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

    # Load encoder and decoder weights
    if pre_train_model is not None:
        load_jepa_encoder(pre_train_model, encoder)
    if r_file is not None:
        load_checkpoint(r_file, decoder)

    unsupervised_loader = iter(unsupervised_loader)

    os.makedirs(folder, exist_ok=True)

    # Sample a batch of clips from the data-loader
    udata = next(unsupervised_loader)
    clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)

    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
        jepa_features = encoder(clips)
        jepa_features = F.layer_norm(jepa_features, (jepa_features.size(-1),))
        pred = decoder(jepa_features)

        reconstructed = unpatchify(
            pred, crop_size, num_frames, patch_size, tubelet_size
        )

    # Unnormalize the original and reconstructed clips for visualization
    clips_unnormalized = unnormalize_tensor(clips, normalize_mean, normalize_std)
    reconstructed_unnormalized = unnormalize_tensor(
        reconstructed, normalize_mean, normalize_std
    )

    # Combine the original and reconstructed clips side by side for comparison
    side_by_side_imgs = torch.cat(
        [clips_unnormalized, reconstructed_unnormalized], dim=4
    )
    for i in range(side_by_side_imgs.shape[0]):
        save_tensor_as_gif(side_by_side_imgs[i], os.path.join(folder, f"{i:02d}.gif"))


if __name__ == "__main__":
    main()
