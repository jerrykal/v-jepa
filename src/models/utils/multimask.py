# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class MultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks=None):
        if masks is None:
            return self.backbone(x)

        if (masks is not None) and not isinstance(masks, list):
            masks = [masks]
        outs = []
        for m in masks:
            outs += [self.backbone(x, masks=m)]
        return outs
    
class PredictorMultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt):
        if type(ctxt) is not list:
            ctxt = [ctxt]
        if type(tgt) is not list:
            tgt = [tgt]
        if type(masks_ctxt) is not list:
            masks_ctxt = [masks_ctxt]
        if type(masks_tgt) is not list:
            masks_tgt = [masks_tgt]

        outs = []
        fully_outs = []
        for i, (zi, hi, mc, mt) in enumerate(zip(ctxt, tgt, masks_ctxt, masks_tgt)):
            prediect_z, fully_z = self.backbone(zi, hi, mc, mt, mask_index=i)
            outs += [prediect_z]
            fully_outs += [fully_z]

        return outs, fully_outs

class WorldModelPredictorMultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def next_frame(self, ctxt, masks_ctxt, masks_tgt, encoded_action):
        if type(ctxt) is not list:
            ctxt = [ctxt]
        if type(masks_ctxt) is not list:
            masks_ctxt = [masks_ctxt]
        if type(masks_tgt) is not list:
            masks_tgt = [masks_tgt]
        outs = []
        completed_outs = []
        for i, (zi, mc, mt) in enumerate(zip(ctxt, masks_ctxt, masks_tgt)):
            prediect_z, completed_z = self.backbone.next_frame(zi, mc, mt, encoded_action, mask_index=-1)
            outs += [prediect_z]
            completed_outs += [completed_z]
        return outs, completed_outs
    
    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt, encoded_action):
        if type(ctxt) is not list:
            ctxt = [ctxt]
        if type(tgt) is not list:
            tgt = [tgt]
        if type(masks_ctxt) is not list:
            masks_ctxt = [masks_ctxt]
        if type(masks_tgt) is not list:
            masks_tgt = [masks_tgt]

        outs = []
        completed_outs = []
        for i, (zi, hi, mc, mt) in enumerate(zip(ctxt, tgt, masks_ctxt, masks_tgt)):
            prediect_z, completed_z = self.backbone(zi, hi, mc, mt, encoded_action, mask_index=i)
            outs += [prediect_z]
            completed_outs += [completed_z]

        return outs, completed_outs
