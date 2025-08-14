import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from app.diffusion_decoder.utils import init_models, load_checkpoint, load_jepa_encoder
from app.vjepa.transforms import make_transforms
from src.datasets.data_manager import init_data
from src.models.diffusion_decoder import JEPADecoderPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def unnormalize_tensor(tensor, mean, std):
    tensor = tensor.clone().contiguous()
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)

    C, T, H, W = tensor.shape
    tensor = tensor.view(C, -1).permute(1, 0)
    tensor.mul_(std).add_(mean)
    tensor = tensor.permute(1, 0).view(C, T, H, W)
    return tensor


def save_tensor_as_gif(tensor, output_path, duration=100):
    """
    Save a tensor as a GIF file.
    """
    # Convert from tensor to numpy and scale to 0-255
    tensor_np = tensor.cpu().numpy().astype(np.uint8)

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
    logger.info(f"Saved video to {output_path}")


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
    save_dir = cfgs_meta.get("save_dir")
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

    # -- DIFFUSION
    cfgs_diffusion = configs.get("diffusion")
    in_channels = cfgs_diffusion.get("in_channels", 3)
    out_channels = cfgs_diffusion.get("out_channels", 3)
    layers_per_block = cfgs_diffusion.get("layers_per_block", 2)
    block_out_channels = cfgs_diffusion.get("block_out_channels", (64, 128, 256, 256))
    down_block_types = cfgs_diffusion.get(
        "down_block_types",
        (
            "DownBlock2D",
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
    )
    up_block_types = cfgs_diffusion.get(
        "up_block_types",
        (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    scheduler_beta_start = cfgs_diffusion.get("scheduler_beta_start", 0.00085)
    scheduler_beta_end = cfgs_diffusion.get("scheduler_beta_end", 0.012)
    scheduler_beta_schedule = cfgs_diffusion.get(
        "scheduler_beta_schedule", "scaled_linear"
    )
    scheduler_prediction_type = cfgs_diffusion.get(
        "scheduler_prediction_type", "epsilon"
    )

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

    # -- DATA AUGS
    cfgs_data_aug = configs.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- INFERENCE
    cfgs_inference = configs.get("inference")
    num_inference_steps = cfgs_inference.get("num_inference_steps", 50)
    num_samples_to_generate = cfgs_inference.get("num_samples_to_generate", 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    # Make data transforms
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
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
        transform=transform,
        datasets_weights=datasets_weights,
        collator=None,
        num_workers=num_workers,
        pin_mem=pin_mem,
    )
    logger.info("Initialized data-loaders/samplers")

    # Init models
    encoder, unet, noise_scheduler = init_models(
        device=device,
        uniform_power=uniform_power,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        use_sdpa=use_sdpa,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        scheduler_beta_start=scheduler_beta_start,
        scheduler_beta_end=scheduler_beta_end,
        scheduler_beta_schedule=scheduler_beta_schedule,
        scheduler_prediction_type=scheduler_prediction_type,
    )
    logger.info("Initialized models")

    # Load encoder weight
    encoder = load_jepa_encoder(pre_train_model, encoder)

    # Load denoising UNet weight
    if r_file is not None:
        unet, noise_scheduler, _, _, _ = load_checkpoint(
            r_path=r_file,
            unet=unet,
            noise_scheduler=noise_scheduler,
        )

    # Create diffusion pipeline
    pipeline = JEPADecoderPipeline(unet=unet, scheduler=noise_scheduler)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    logger.info("Created diffusion pipeline")

    unsupervised_loader = iter(unsupervised_loader)

    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(num_samples_to_generate), desc="Generating samples"):
        # Sample one clip from the data-loader
        udata = next(unsupervised_loader)
        clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)

        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
            # Encode the clip into JEPA features
            encoder_hidden_states = encoder(clips)
            encoder_hidden_states = F.layer_norm(
                encoder_hidden_states, (encoder_hidden_states.size(-1),)
            )

            N_t = num_frames // tubelet_size
            encoder_hidden_states = rearrange(
                encoder_hidden_states, "b (t n) d -> b t n d", t=N_t
            )

            # Repeat the encoder hidden states for each tubelet
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                repeats=tubelet_size, dim=1
            )

            # Reshape the encoder output to be [B * T, N, D]
            encoder_hidden_states = rearrange(
                encoder_hidden_states, "b t n d -> (b t) n d"
            )

            # Create class labels indicating which frame in the tubelet we are reconstructing
            class_labels = torch.arange(tubelet_size, device=device).repeat(
                encoder_hidden_states.shape[0] // tubelet_size
            )

            reconstructed_images = pipeline(
                encoder_hidden_states=encoder_hidden_states,
                class_labels=class_labels,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=device).manual_seed(seed),
            )

        # Rearrange both clips and reconstructed images to be (C, T, H, W)
        clips = rearrange(clips, "b c t h w -> c (b t) h w")
        reconstructed_images = reconstructed_images.permute(1, 0, 2, 3)

        # Unnormalize the clips and reconstructed images for visualization
        clips_unnormalized = unnormalize_tensor(clips, transform.mean, transform.std)
        reconstructed_unnormalized = unnormalize_tensor(
            reconstructed_images, transform.mean, transform.std
        )

        # Save both original and reconstructed images side by side
        combined_images = torch.cat(
            [clips_unnormalized, reconstructed_unnormalized], dim=3
        )
        save_tensor_as_gif(
            combined_images,
            os.path.join(save_dir, f"{i:02d}.gif"),
        )


if __name__ == "__main__":
    main()
