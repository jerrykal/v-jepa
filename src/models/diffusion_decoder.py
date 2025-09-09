import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    EDMEulerScheduler,
    UNet2DConditionModel,
    UNet2DModel,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from einops import rearrange


class JEPADecoderPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler | EDMEulerScheduler,
        vae: AutoencoderKL | None = None,
        img_size: int = 224,
        patch_size: int = 16,
        cross_attn_cond: bool = True,
        in_concat_cond: bool = False,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.cross_attn_cond = cross_attn_cond
        self.in_concat_cond = in_concat_cond
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae)

    @torch.no_grad()
    def __call__(
        self,
        jepa_features: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        num_inference_steps: int = 50,
        generator: torch.Generator | None = None,
    ):
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)
        timesteps = self.scheduler.timesteps

        # Prepare noisy image
        latents_shape = (
            1,
            self.unet.config.in_channels,
            self.unet.config.sample_size,
            self.unet.config.sample_size,
        )
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self._execution_device,
            dtype=jepa_features.dtype,
        )

        # Repeat the noise so that each image in the clip is denoised from the same noise
        latents = latents.repeat(jepa_features.shape[0], 1, 1, 1)

        if self.in_concat_cond:
            jepa_cond = jepa_features.clone()
            jepa_cond = rearrange(
                jepa_cond,
                "b (t p1 p2) d -> b (t d) p1 p2",
                p1=self.img_size // self.patch_size,
                p2=self.img_size // self.patch_size,
            )
            jepa_cond = F.interpolate(
                jepa_cond, size=(latents.shape[2], latents.shape[3]), mode="bilinear"
            )

            latents[:, : -self.unet.config.out_channels] = jepa_cond

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t in timesteps:
                latent_model_input = self.scheduler.scale_model_input(latents, t)

                if self.cross_attn_cond:
                    noise_pred = self.unet(
                        latent_model_input,
                        timestep=t,
                        encoder_hidden_states=jepa_features,
                        class_labels=class_labels,
                        return_dict=False,
                    )[0]
                else:
                    noise_pred = self.unet(
                        latent_model_input,
                        timestep=t,
                        class_labels=class_labels,
                        return_dict=False,
                    )[0]

                if self.in_concat_cond:
                    frame_latents = latents[:, -self.unet.config.out_channels :]
                    frame_latents = self.scheduler.step(
                        noise_pred, t, frame_latents, return_dict=False
                    )[0]
                    latents[:, -self.unet.config.out_channels :] = frame_latents
                else:
                    latents = self.scheduler.step(
                        noise_pred, t, latents, return_dict=False
                    )[0]

                progress_bar.update()

        if self.in_concat_cond:
            latents = latents[:, -self.unet.config.out_channels :]

        if self.vae is not None:
            generated_images = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=True
            )[0]
        else:
            generated_images = latents

        return generated_images


def get_unet_and_scheduler(
    sample_size: int,
    in_channels: int,
    out_channels: int,
    layers_per_block: int,
    attention_head_dim: int,
    dropout: float,
    block_out_channels: tuple[int, ...],
    down_block_types: tuple[str, ...],
    up_block_types: tuple[str, ...],
    num_class_embeds: int,
    cross_attention_dim: int | None = None,
    scheduler_beta_start: float = 0.00085,
    scheduler_beta_end: float = 0.012,
    scheduler_beta_schedule: str = "scaled_linear",
    scheduler_prediction_type: str = "epsilon",
    do_edm_style_training: bool = False,
) -> tuple[UNet2DConditionModel | UNet2DModel, DDIMScheduler | EDMEulerScheduler]:
    if cross_attention_dim is not None:
        unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            cross_attention_dim=cross_attention_dim,
            num_class_embeds=num_class_embeds,
        )
    else:
        unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            num_class_embeds=num_class_embeds,
        )
    if do_edm_style_training:
        noise_scheduler = EDMEulerScheduler(prediction_type=scheduler_prediction_type)
    else:
        noise_scheduler = DDIMScheduler(
            beta_start=scheduler_beta_start,
            beta_end=scheduler_beta_end,
            beta_schedule=scheduler_beta_schedule,
            prediction_type=scheduler_prediction_type,
        )

    return unet, noise_scheduler
