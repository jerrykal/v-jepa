import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import src.models.vision_transformer as video_vit
from src.models.attentive_pooler import AttentivePooler
from src.models.utils.multimask import MultiMaskWrapper
from src.utils.logging import get_logger

from src.utils.tensors import normalize_tensor

logger = get_logger(__name__)

def _strip_module_prefix(sd):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def _load_component(checkpoint, name, model, use_ddp):
    if model is not None and name in checkpoint:
        try:
            ckpt= checkpoint[name] if use_ddp else _strip_module_prefix(checkpoint[name])
            msg = model.load_state_dict(ckpt)
            logger.info(f'Loaded {name} with msg: {msg}')
        except Exception as e:
            logger.warning(f'Failed to load {name}: {e}')
    else:
        logger.warning(f'No "{name}" found in checkpoint.')
    return model

def load_pretrained_model(
    model_path,
    encoder,
    gradient=False,
    use_ddp=False
):
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint: {e}')
        return encoder

    try:
        module = _load_component(checkpoint, 'target_encoder', encoder, use_ddp)
        if not gradient:
            for param in module.parameters():
                param.requires_grad_(False)

    except Exception as e:
        logger.info(f'Failed to load models from checkpoint: {e}')

    return encoder

class JEPAExtractor(BaseFeaturesExtractor):
    """
    Simple CNN feature extractor with customizable conv layers.
    conv_spec: List of (out_channels, kernel_size, stride, padding).
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        model_name: str = "vit_h",
        pretrain_path: str|None = None,
        crop_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        uniform_power: bool = False,
        use_sdpa: bool = True,
        pooler_params: dict = {}
    ):
        super().__init__(observation_space, features_dim)
        encoder =  video_vit.__dict__[model_name](
            img_size=crop_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            uniform_power=uniform_power,
            use_sdpa=use_sdpa, # not using
        )
        encoder = MultiMaskWrapper(encoder)
        self._encoder = load_pretrained_model(pretrain_path, encoder, False)
        
        self._pooler = AttentivePooler(
            num_queries=1,
            embed_dim=encoder.backbone.embed_dim,
            num_heads=pooler_params["num_heads"],
            mlp_ratio=pooler_params["mlp_ratio"],
            depth=pooler_params["depth"],
            norm_layer=nn.SiLU,
            init_std=pooler_params["init_std"],
            qkv_bias=pooler_params["qkv_bias"],
            complete_block=pooler_params["complete_block"],
        )

        self._out_linear = nn.Linear(encoder.backbone.embed_dim, features_dim, bias=True)



    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B, CT, H, W = observations.shape 
        with torch.amp.autocast(device_type=observations.device.type, dtype=torch.bfloat16, enabled=True):
            observations = observations.view(B, 3, 4, H, W)
            observations = observations.repeat_interleave(4, dim=2)
            x = normalize_tensor(observations)
            x = self._encoder(x)
            x = F.layer_norm(x, (x.size(-1),))
            x = self._pooler(x).squeeze(1)
            x = self._out_linear(x)
        return x