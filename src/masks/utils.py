# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)

def apply_masks_ragged(x, masks, concat=True):
    B, N, D = x.shape
    result = []

    for m in masks:  # mask_type: List[Tensor], length = B
        masked_samples = []
        for b_idx, mask in enumerate(m):
            selected = x[b_idx].index_select(0, mask)  # shape [K_i, D]
            masked_samples.append(selected)
        result.append(masked_samples)

    if not concat:
        return result

    flat_result = [sample for mask_type in result for sample in mask_type]
    return torch.cat(flat_result, dim=0)

def apply_masks_skip_action(x, masks, patch_per_frame, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    masks_mapped = map_patch_indices_to_with_action(masks, patch_per_frame)

    all_x = []
    for m in masks_mapped:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x

    return torch.cat(all_x, dim=0)

def map_patch_indices_to_with_action(masks, patch_per_frame):
    mapped = []
    for m in masks:
        frame_idx = m // patch_per_frame
        patch_idx = m % patch_per_frame
        new_idx = frame_idx * (patch_per_frame + 1) + patch_idx
        mapped.append(new_idx)
    return mapped
