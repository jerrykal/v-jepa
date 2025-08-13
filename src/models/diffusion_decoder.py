import torch
from diffusers import PNDMScheduler, UNet2DConditionModel, UNet2DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


class JEPADecoderPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet: UNet2DConditionModel | UNet2DModel,
        scheduler: PNDMScheduler,
    ):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        jepa_features: torch.Tensor | None = None,
        num_inference_steps: int = 50,
        generator: torch.Generator | None = None,
    ):
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self._execution_device)
        timesteps = self.scheduler.timesteps

        # Prepare noisy image
        image_shape = (
            1,
            3,
            self.unet.config.sample_size,
            self.unet.config.sample_size,
        )
        noisy_images = torch.randn(
            image_shape,
            generator=generator,
            device=self._execution_device,
            dtype=jepa_features.dtype,
        )

        # Repeat the noise so that each image in the clip is denoised from the same noise
        noisy_images = noisy_images.repeat(jepa_features.shape[0], 1, 1, 1)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t in timesteps:
                model_input = self.scheduler.scale_model_input(noisy_images, t)

                if isinstance(self.unet, UNet2DConditionModel):
                    noise_pred = self.unet(
                        model_input,
                        timestep=t,
                        encoder_hidden_states=jepa_features,
                        return_dict=False,
                    )[0]
                else:
                    noise_pred = self.unet(
                        model_input,
                        timestep=t,
                        return_dict=False,
                    )[0]

                noisy_images = self.scheduler.step(
                    noise_pred, t, noisy_images, return_dict=False
                )[0]

                progress_bar.update()

        generated_images = noisy_images
        return generated_images


def get_unet_and_scheduler(
    sample_size: int,
    in_channels: int,
    out_channels: int,
    layers_per_block: int,
    block_out_channels: tuple[int, ...],
    down_block_types: tuple[str, ...],
    up_block_types: tuple[str, ...],
    cross_attention_dim: int,
    scheduler_beta_start: float = 0.00085,
    scheduler_beta_end: float = 0.012,
    scheduler_beta_schedule: str = "scaled_linear",
    scheduler_prediction_type: str = "epsilon",
):
    unet = UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        cross_attention_dim=cross_attention_dim,
    )
    noise_scheduler = PNDMScheduler(
        beta_start=scheduler_beta_start,
        beta_end=scheduler_beta_end,
        beta_schedule=scheduler_beta_schedule,
        prediction_type=scheduler_prediction_type,
    )

    return unet, noise_scheduler
