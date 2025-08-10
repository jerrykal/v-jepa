from diffusers import PNDMScheduler, UNet2DConditionModel, UNet2DModel


def get_unet_and_scheduler(
    sample_size: int,
    in_channels: int,
    out_channels: int,
    layers_per_block: int,
    block_out_channels: tuple[int, ...],
    down_block_types: tuple[str, ...],
    up_block_types: tuple[str, ...],
    cross_attention_dim: int | None = None,
    jepa_conditioned: bool = True,
    scheduler_beta_start: float = 0.00085,
    scheduler_beta_end: float = 0.012,
    scheduler_beta_schedule: str = "scaled_linear",
    scheduler_prediction_type: str = "epsilon",
):
    assert cross_attention_dim is not None or jepa_conditioned is False, (
        "cross_attention_dim must be provided if jepa_conditioned is True"
    )

    if jepa_conditioned:
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
    else:
        unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )
    noise_scheduler = PNDMScheduler(
        beta_start=scheduler_beta_start,
        beta_end=scheduler_beta_end,
        beta_schedule=scheduler_beta_schedule,
        prediction_type=scheduler_prediction_type,
    )

    return unet, noise_scheduler
