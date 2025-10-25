import torch.nn as nn
from einops import rearrange
from src.models.attentive_pooler import AttentivePooler
from src.models.utils.modules import SpatialAttention, TemporalAttention
from src.models.utils.quantization import VectorQuantization
from torch import Tensor


class LatentActionEncoder(nn.Module):
    """
    Latent Action Model (LAM) used to distill latent actions
    from history of past video frames. The LAM model employs a
    VQ-VAE model to encode video frames into discrete latents.
    Both the encoder and decoder are based on spatial-temporal
    transformers.
    """

    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        num_patches_per_frame: int,
        d_codebook: int,
        n_codebook: int,
        vq_bias: bool = True,
        vq_commit_weight: float = 0.25,
        vq_entropy_weight: float = 0.1,
        vq_diversity_weight: float = 1.0,
        quant_loss_weight: float = 1.0,
        use_sdpa=True,
    ) -> None:
        super().__init__()

        self.enc_layer = nn.ModuleList(
            [
                SpatialAttention(
                    dims=input_dim,
                    num_heads=num_heads,
                    qkv_bias=False,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    use_sdpa=use_sdpa,
                ),
                TemporalAttention(
                    dims=input_dim,
                    num_heads=num_heads,
                    qkv_bias=False,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    use_sdpa=use_sdpa,
                    is_causal=True,
                ),
            ]
        )

        self.attentive_pooler = AttentivePooler(
            num_queries=1,
            embed_dim=input_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            depth=2,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            qkv_bias=True,
            complete_block=True,
        )

        self.quant = VectorQuantization(
            input_dim=input_dim,
            embedding_dim=d_codebook,
            num_embeddings=n_codebook,
            commitment_cost=vq_commit_weight,
        )

        self.d_codebook = d_codebook
        self.n_codebook = n_codebook
        self.quant_loss_weight = quant_loss_weight

    def encode(self, x: Tensor) -> tuple[Tensor, dict]:
        """
        Args:
            x: Tensor of shape [B, T, P, D], pre-encoded video features
        Returns:
            quantized latent actions: Tensor of shape [B, T, D]
            loss_dict: dict of individual losses
        """

        # 1. Spatial + Temporal Attention
        for block in self.enc_layer:
            h = block(x)  # [B, T, P, D]

        # 2. Apply attentive pooler: [B, T, P, D] -> [B, T, 1, D]
        bsz = h.size(0)
        h = rearrange(h, "b t p d -> (b t) p d")
        h = self.attentive_pooler(h)  # [B * T, 1, D]
        h = rearrange(h, "(b t) 1 d -> b t 1 d", b=bsz)

        # 3. Quantize latent action
        (z_q, _), q_loss = self.quant(h)  # z_q: [B, T, 1, D]

        loss = (q_loss * self.quant_loss_weight) if self.training else 0
        return z_q, loss

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        """
        Args:
            x: Tensor of shape [B, T, P, D]
        Returns:
            quantized_actions: [B, P*T, D_q] or similar
            loss_dict: dict of losses (used in training)
        """
        return self.encode(x)
