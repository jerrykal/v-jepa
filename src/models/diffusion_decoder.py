import torch
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel, UNet2DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


class JEPADecoderPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
        vae: AutoencoderKL | None = None,
        img_size: int = 224,
    ):
        super().__init__()

        self.img_size = img_size
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae)

    @torch.no_grad()
    def __call__(
        self,
        encoder_hidden_states: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        num_inference_steps: int = 50,
        generator: torch.Generator | None = None,
    ):
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)
        timesteps = self.scheduler.timesteps

        # Prepare noisy image
        downsample_factor = self.vae.config.sample_size // self.unet.config.sample_size
        latents_shape = (
            1,
            self.unet.config.in_channels,
            self.img_size // downsample_factor,
            self.img_size // downsample_factor,
        )
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self._execution_device,
            dtype=encoder_hidden_states.dtype,
        )

        # Repeat the noise so that each image in the clip is denoised from the same noise
        latents = latents.repeat(encoder_hidden_states.shape[0], 1, 1, 1)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t in timesteps:
                latent_model_input = self.scheduler.scale_model_input(latents, t)

                noise_pred = self.unet(
                    latent_model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    class_labels=class_labels,
                    return_dict=False,
                )[0]
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                progress_bar.update()

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
    block_out_channels: tuple[int, ...],
    down_block_types: tuple[str, ...],
    up_block_types: tuple[str, ...],
    num_class_embeds: int,
    cross_attention_dim: int | None = None,
    scheduler_beta_start: float = 0.00085,
    scheduler_beta_end: float = 0.012,
    scheduler_beta_schedule: str = "scaled_linear",
    scheduler_prediction_type: str = "epsilon",
):
    if cross_attention_dim is not None:
        unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
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
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            num_class_embeds=num_class_embeds,
        )
    noise_scheduler = PNDMScheduler(
        beta_start=scheduler_beta_start,
        beta_end=scheduler_beta_end,
        beta_schedule=scheduler_beta_schedule,
        prediction_type=scheduler_prediction_type,
    )

    return unet, noise_scheduler
