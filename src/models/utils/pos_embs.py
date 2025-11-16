# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np


def get_3d_sincos_pos_embed(
    embed_dim,
    grid_size,
    grid_depth,
    cls_token=False,
    uniform_power=False,
    action_tokens=False,
):
    """
    grid_size: int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(
        grid_h, grid_d, grid_w
    )  # order of meshgrid is very important for indexing as [d,h,w]

    if not uniform_power:
        embed_dim_h = embed_dim // 4
        embed_dim_w = embed_dim // 4
        embed_dim_d = embed_dim // 2
    else:
        embed_dim_h = embed_dim_w = embed_dim_d = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim_h, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim_w, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim_d, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]

    if action_tokens:
        # For action tokens, assign only temporal (depth) positional embeddings; spatial embeddings are set to zero.
        grid_d_act = np.arange(grid_depth, dtype=float)
        emb_h_act = np.zeros([grid_depth, embed_dim_h], dtype=float)
        emb_w_act = np.zeros([grid_depth, embed_dim_w], dtype=float)
        emb_d_act = get_1d_sincos_pos_embed_from_grid(embed_dim_d, grid_d_act)
        pos_embed_act = np.concatenate([emb_d_act, emb_h_act, emb_w_act], axis=1)

        # Interleave action token between each frames
        pos_embed = pos_embed.reshape(grid_depth, grid_size * grid_size, embed_dim)  # [T, H*W, D]
        pos_embed_act = pos_embed_act[:, None, :]  # [T, 1, D]
        pos_embed = np.concatenate([pos_embed_act, pos_embed], axis=1)
        pos_embed = pos_embed.reshape(-1, embed_dim)  # [T * (1+H*W), D]

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, action_token=False):
    """
    grid_size: int of the grid height and width
    returns:
        pos_embed: [grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    if action_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
