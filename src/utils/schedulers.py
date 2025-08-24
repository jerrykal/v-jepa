# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from typing import Dict, Any

class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = float(start_lr)
        self.ref_lr = float(ref_lr)
        self.final_lr = float(final_lr)
        self.warmup_steps = int(warmup_steps)


        self.total_steps = int(T_max)
        self.T_max = max(1, self.total_steps - self.warmup_steps)

        self._step = 0.

        self._apply_current_lr()


    def _lr_at(self, step: float) -> float:
        if step < self.warmup_steps:
            progress = float(step) / float(max(1, self.warmup_steps))
            return self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = float(step - self.warmup_steps) / float(max(1, self.T_max))
            return max(
                self.final_lr,
                self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )

    def _apply_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def _apply_current_lr(self) -> float:
        lr = self._lr_at(self._step)
        self._apply_lr(lr)
        return lr

    def step(self) -> float:
        self._step += 1.0
        return self._apply_current_lr()
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "start_lr": self.start_lr,
            "ref_lr": self.ref_lr,
            "final_lr": self.final_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,  
            "T_max_internal": self.T_max,    
            "step": self._step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:

        self.start_lr = float(state_dict.get("start_lr", self.start_lr))
        self.ref_lr = float(state_dict.get("ref_lr", self.ref_lr))
        self.final_lr = float(state_dict.get("final_lr", self.final_lr))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.T_max = max(1, self.total_steps - self.warmup_steps)

        self._step = float(state_dict.get("step", self._step))
        self.last_epoch = int(state_dict.get("last_epoch", self.last_epoch))


        self._apply_current_lr()

    def __repr__(self) -> str:
        return (
            f"WarmupCosineSchedule(start_lr={self.start_lr}, ref_lr={self.ref_lr}, "
            f"final_lr={self.final_lr}, warmup_steps={self.warmup_steps}, total_steps={self.total_steps}, "
            f"step={self._step})"
        )

class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd: float,
        T_max: int,
        final_wd: float = 0.0,
    ):
        self.optimizer = optimizer
        self.ref_wd = float(ref_wd)
        self.final_wd = float(final_wd)


        self.T_max = max(1, int(T_max))

        self._step = 0.0

        self._apply_current_wd()

    def _wd_at(self, step: float) -> float:
        progress = float(step) / float(self.T_max)
        wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
        if self.final_wd <= self.ref_wd:
            wd = max(self.final_wd, wd)
        else:
            wd = min(self.final_wd, wd)
        return wd

    def _apply_wd(self, wd: float) -> None:
        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or (not group["WD_exclude"]):
                group["weight_decay"] = float(wd)

    def _apply_current_wd(self) -> float:
        wd = self._wd_at(self._step)
        self._apply_wd(wd)
        return wd
    
    def step(self) -> float:
        self._step += 1.0
        return self._apply_current_wd()
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "ref_wd": self.ref_wd,
            "final_wd": self.final_wd,
            "T_max": self.T_max,
            "step": self._step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.ref_wd = float(state_dict.get("ref_wd", self.ref_wd))
        self.final_wd = float(state_dict.get("final_wd", self.final_wd))
        self.T_max = max(1, int(state_dict.get("T_max", self.T_max)))
        self._step = float(state_dict.get("step", self._step))
        
        self._apply_current_wd()

    def __repr__(self) -> str:
        return (
            f"CosineWDSchedule(ref_wd={self.ref_wd}, final_wd={self.final_wd}, "
            f"T_max={self.T_max}, step={self._step})"
        )