import torch.nn as nn

from typing import Tuple
from torch import Tensor
from math import prod
from torch.nn.functional import mse_loss
from einops.layers.torch import Rearrange
from src.models.utils.modules import SpatialAttention, TemporalAttention
from src.models.attentive_pooler import AttentivePooler
from src.models.utils.quantization import LookupFreeQuantization
from src.models.predictor import vit_predictor

REPR_ACT_ENC = (
    ('space-time_attn', {
        'n_repr' : 8,
        'n_heads': 8,
        'd_head': 64,
    }),
)


class LatentActionEncoder(nn.Module):
    '''
        Latent Action Model (LAM) used to distill latent actions
        from history of past video frames. The LAM model employs a
        VQ-VAE model to encode video frames into discrete latents.
        Both the encoder and decoder are based on spatial-temporal
        transformers.
    '''
    def __init__(
        self,
        num_heads: int,
        d_codebook: int,
        inp_dims: int = 192, 
        n_codebook: int = 1,
        lfq_bias: bool = True,
        lfq_commit_weight: float = 0.25,
        lfq_entropy_weight: float = 0.1,
        lfq_diversity_weight: float = 1.,
        quant_loss_weight: float = 1.,
    ) -> None:   
        super().__init__()

        self.enc_layer = nn.ModuleList([
            SpatialAttention(
                dims=inp_dims,
                num_heads=num_heads,
                qkv_bias=False,
                qk_scale=None,
                drop=0.,
                attn_drop=0.
            ),
            TemporalAttention(
                dims=inp_dims,
                num_heads=num_heads,
                qkv_bias=False,
                qk_scale=None,
                drop=0.,
                attn_drop=0.
            )
        ])
        
        self.attenion_pooler = AttentivePooler(
            num_queries=1,
            embed_dim=inp_dims,
            num_heads=num_heads,
            mlp_ratio=4.0,
            depth=2,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            qkv_bias=True,
            complete_block=True,
        )

        # # Add the projections to the action space
        # self.to_act = nn.Sequential(
        #         Rearrange('b c t ... -> b t (c ...)'),
        #         nn.Linear(
        #             int(n_embd),
        #             d_codebook,
        #             bias=False,
        #         )
        # )

        # Build the quantization module
        self.quant = LookupFreeQuantization(
            input_dim           = inp_dims,
            codebook_dim        = d_codebook,
            num_codebook        = n_codebook,
            use_bias            = lfq_bias,
            commit_weight       = lfq_commit_weight,
            entropy_weight      = lfq_entropy_weight,
            diversity_weight    = lfq_diversity_weight,
        )
        
        self.d_codebook = d_codebook
        self.n_codebook = n_codebook
        self.quant_loss_weight = quant_loss_weight

    def encode(self, x: Tensor) -> Tuple[Tensor, dict]:
        '''
        Args:
            x: Tensor of shape [B, T, P, D], pre-encoded video features
        Returns:
            quantized latent actions: Tensor of shape [B, T, D_q]
            loss_dict: dict of individual losses
        '''

        # === 1. Spatial + Temporal Attention ===
        for block in self.enc_layer:
            x = block(x)  # [B, T, P, D]

        B, T, P, D = x.shape

        # === 2. Flatten time and space: [B, T, P, D] → [B, T*P, D]
        x = x.permute(0, 2, 1, 3).reshape(B, T * P, D)

        # === 3. Apply Attentive Pooler: [B, T*P, D] → [B, 1, D]
        pooled = self.attenion_pooler(x)  # returns [B, 1, D]
        pooled = pooled.squeeze(1)        # [B, D]

        # === 4. Quantize
        (z_q, idx), q_loss = self.quant(pooled)  # z_q: [B, D_q]

        # === 5. Output and loss
        loss = q_loss * self.quant_loss_weight if self.training and q_loss is not None else 0
        
        return z_q, loss

    def forward(self, x: Tensor) -> Tuple[Tensor, dict]:
        '''
        Args:
            x: Tensor of shape [B, T, P, D]
        Returns:
            quantized_actions: [B, P*T, D_q] or similar
            loss_dict: dict of losses (used in training)
        '''
        return self.encode(x)
    