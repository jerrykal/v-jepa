# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial

import torch
import torch.nn as nn
from src.models.utils.modules import Block, build_action_block_causal_attention_mask
from src.models.utils.pos_embs import get_3d_sincos_pos_embed
from src.utils.tensors import trunc_normal_


class VisionTransformerPredictorAC(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 1,
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        action_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        init_std: float = 0.02,
        uniform_power: bool = False,
        is_causal: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Map input to predictor dimension
        self.predictor_embed_context = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.predictor_embed_action = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)

        # Determine positional embedding
        self.input_size = img_size
        self.patch_size = patch_size

        # In the action-conditioned setup, each video frame is encoded separately.
        # Therefore, unlike the standard JEPA configuration, there is no temporal compression applied.
        self.num_frames = num_frames
        self.num_patches = num_patches = num_frames * (img_size // patch_size) * (img_size // patch_size)

        # Position embedding
        self.uniform_power = uniform_power
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames + num_patches, predictor_embed_dim), requires_grad=False
        )

        # Attention Blocks
        self.grid_size = self.input_size // self.patch_size
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.GELU,
                    attn_drop=attn_drop_rate,
                    grid_size=self.grid_size,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Initialize attention mask
        self.attn_mask = None
        if is_causal:
            self.attn_mask = build_action_block_causal_attention_mask(
                self.num_frames, self.grid_size, self.grid_size, add_tokens=1
            )

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ------ initialize weights
        if self.predictor_pos_embed is not None:
            self._init_pos_embed(self.predictor_pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed: torch.Tensor) -> None:
        embed_dim = pos_embed.size(-1)
        sincos = get_3d_sincos_pos_embed(
            embed_dim,
            self.grid_size,
            self.num_frames,
            cls_token=False,
            uniform_power=self.uniform_power,
            action_tokens=True,
        )
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self) -> None:
        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, contexts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Map tokens to predictor dimensions
        contexts = self.predictor_embed_context(contexts)
        actions = self.predictor_embed_action(actions)

        B, T, _, D = contexts.size()

        # Interleave action token between each frames
        x = torch.cat([actions, contexts], dim=2).flatten(1, 2)  # [B, T*(1+H*W), D]

        attn_mask = self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)

        # Fwd prop
        for blk in self.predictor_blocks:
            x = blk(x, attn_mask=attn_mask)

        # Remove action tokens
        x = x.view(B, T, -1, D)
        x = x[:, :, 1:, :].flatten(1, 2)  # [B, T*H*W, D]

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        return x


def vit_ac_predictor(**kwargs) -> VisionTransformerPredictorAC:
    model = VisionTransformerPredictorAC(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
