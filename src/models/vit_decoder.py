# ViT based decoder inspired by masked autoencoder: https://github.com/facebookresearch/mae/blob/main/models_mae.py

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from src.models.utils.pos_embs import get_3d_sincos_pos_embed


class ViTVideoDecoder(nn.Module):
    """
    ViT-based decoder for reconstructing video from V-JEPA encoded representations.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        in_dim: int = 1280,
        embed_dim: int = 640,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        """
        Args:
            img_size: Size of the input image.
            patch_size: Size of the input patches.
            num_frames: Number of frames in the input video.
            tubelet_size: Size of the tubelet.
            in_channels: Number of channels in the input video.
            in_dim: Dimension of the input V-JEPA representations.
            embed_dim: Dimension of the hidden representations in the transformer blocks.
            depth: Depth of the transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of the MLP hidden dimension to the embedding dimension.
            norm_layer: Normalization layer.
        """
        super().__init__()

        num_patches = (num_frames // tubelet_size) * (img_size // patch_size) ** 2
        self.in_proj = (
            nn.Linear(in_dim, embed_dim) if in_dim != embed_dim else nn.Identity()
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.out_proj = nn.Linear(
            embed_dim, tubelet_size * (patch_size**2) * in_channels, bias=True
        )

        # Initialize weights
        self._init_pos_embed(
            embed_dim, img_size // patch_size, num_frames // tubelet_size
        )
        self.apply(self._init_weights)

    def _init_pos_embed(self, embed_dim: int, grid_size: int, grid_depth: int) -> None:
        sin_cos = get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth)
        self.pos_embed.data.copy_(torch.from_numpy(sin_cos).float().unsqueeze(0))

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: A 3D tensor of shape (B, L, D) representing a batch of V-JEPA representations.
               - B: batch size
               - L: number of patches
               - D: dimension of V-JEPA representations

        Returns:
            A tensor of shape (B, L, T * P * P * C) where each element is a reconstructed patch.
               - B: batch size
               - L: number of patches
               - T: number of tubelets
               - P: patch size
               - C: number of channels in the output video
        """
        x = self.in_proj(x)

        # Add positional embeddings
        x = x + self.pos_embed

        # Apply transformer blocks & normalize
        x = self.blocks(x)
        x = self.norm(x)

        x = self.out_proj(x)
        return x
