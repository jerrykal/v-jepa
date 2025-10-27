# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from math import pi
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import Tensor


def build_action_block_causal_attention_mask(T, H, W, add_tokens=1):
    N_T = add_tokens + (H * W)
    N = T * N_T
    mask = torch.zeros(N, N).bool()
    mask_block = torch.ones(N_T, N_T).bool()
    local_window_time = T

    for t1 in range(T):
        for t2 in range(max(0, t1 - local_window_time + 1), t1 + 1):
            mask[t1 * N_T : (t1 + 1) * N_T, t2 * N_T : (t2 + 1) * N_T] = mask_block

    return mask


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        kind: Literal["1d", "2d", "const"] = "1d",
        theta=10000,
        max_freq=10,
        num_freq=1,
        learned_freq=False,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
    ) -> None:
        super().__init__()

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        match kind:
            case "1d":
                freq = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
            case "2d":
                freq = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
            case "const":
                freq = torch.ones(num_freq).float()

        self.freq = nn.Parameter(freq, requires_grad=learned_freq)

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        self.default_seq_dim = -2

    def forward(
        self,
        seq: Tensor,
        seq_dim: int | None = None,
        offset=0,
    ) -> Tensor:
        seq_dim = seq_dim if seq_dim is not None else self.default_seq_dim
        seq_len = seq.shape[seq_dim]

        freq = self.freq

        # Get sequence position
        pos = (torch.arange(seq_len, device=freq.device) + offset) / self.interpolate_factor

        freq = einsum(pos, freq, "..., f -> ... f")
        freq = repeat(freq, "... n -> ... (n r)", r=2)

        if seq_dim == -3:
            freq = rearrange(freq, "n d -> n 1 d")

        # Apply rotary embedding
        return self.apply(freq, seq, seq_dim=seq_dim)

    def apply(self, freq: Tensor, seq: Tensor, start_index: int = 0, scale: float = 1.0, seq_dim: int = -2) -> Tensor:
        dtype = seq.dtype

        if seq.ndim == 3:
            seq_len = seq.shape[seq_dim]
            freq = freq[-seq_len:]

        rot_dim = freq.shape[-1]
        end_index = start_index + rot_dim

        assert rot_dim <= seq.shape[-1], (
            f"feature dimension {seq.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        )

        t_left, seq, t_right = seq[..., :start_index], seq[..., start_index:end_index], seq[..., end_index:]

        seq = (seq * freq.cos() * scale) + (self.rotate_half(seq) * freq.sin() * scale)
        out = torch.cat((t_left, seq, t_right), dim=-1)

        return out.type(dtype)

    def rotate_half(self, inp: Tensor) -> Tensor:
        inp = rearrange(inp, "... (d r) -> ... d r", r=2)
        x1, x2 = inp.unbind(dim=-1)
        inp = torch.stack((-x2, x1), dim=-1)
        return rearrange(inp, "... d r -> ... (d r)")

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x, mask=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if attn_mask is not None or self.is_causal is True or self.use_sdpa:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        grid_size=None,
        grid_depth=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, mask=None, attn_mask=None):
        y, attn = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask)
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = xattn @ v

        q = q.transpose(1, 2).reshape(B, n, C)
        return q


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q


class SpatialAttention(nn.Module):
    def __init__(self, dims, num_heads, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, use_sdpa=True):
        super().__init__()
        self.embed = RotaryEmbedding(dims, kind="1d")

        self.attn = Attention(
            dim=dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_sdpa=use_sdpa,
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, P, D = x.shape
        x = x.view(B * T, P, D)  # [B*T, P, D]
        x = self.embed(x, seq_dim=1)  # rotary along patch dim
        x, _ = self.attn(x)  # attention on spatial patches
        x = x.view(B, T, P, D)  # [B, T, P, D]
        return x


class TemporalAttention(nn.Module):
    def __init__(
        self, dims, num_heads, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, use_sdpa=True, is_causal=False
    ):
        super().__init__()
        self.embed = RotaryEmbedding(dims, kind="1d")

        self.attn = Attention(
            dim=dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_sdpa=use_sdpa,
            is_causal=is_causal,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, P, D]
        B, T, P, D = x.shape
        # Reshape to [B * P, T, D] to group all patches along time
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, P, T, D]
        x = x.view(B * P, T, D)  # [B*P, T, D]
        # Apply rotary embedding along time dimension
        x = self.embed(x, seq_dim=1)
        # Apply attention
        x, _ = self.attn(x)  # [B*P, T, D]
        # Reshape back to [B, T, P, D]
        x = x.view(B, P, T, D).permute(0, 2, 1, 3).contiguous()  # [B, T, P, D]
        return x
