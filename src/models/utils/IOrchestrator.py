from __future__ import annotations
from typing import Dict, Iterable, Any, Optional
import torch
import torch.nn as nn

class IOrchestrator:
    """
    A lightweight training/inference engine that manages lifecycle and resources.
    - Keeps modules as nn.Module, but does not itself subclass nn.Module.
    - Centralizes train/eval/to/params/state_dict/autocast helpers.
    - Subclasses implement `forward_loss(...)` (and optionally `forward_infer(...)`).
    """

    def __init__(
        self,
        modules: Dict[str, nn.Module],
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler] = None,
        schedulers: Optional[Dict[str, Any]] = None,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        clip_grad: Optional[float] = None,
    ):
        self._modules = dict(modules)
        self._optimizer = optimizer
        self._scaler = scaler
        self._schedulers = schedulers or {}
        self._use_amp = use_amp
        self._amp_dtype = amp_dtype
        self._clip_grad = clip_grad

    # ---------- lifecycle ----------
    def modules(self) -> Dict[str, nn.Module]:
        return self._modules
    
    def parameters(self) -> Iterable[torch.nn.Parameter]:
        for m in self._modules.values():
            yield from m.parameters()

    def train(self):
        for m in self._modules.values():
            m.train()

    def eval(self):
        for m in self._modules.values():
            m.eval()

    def to(self, device, dtype=None):
        for m in self._modules.values():
            m.to(device=device, dtype=dtype)
        return self
    
    def device(self) -> torch.device:
        # infer from the first param of the first module
        for m in self._modules.values():
            p = next(m.parameters(), None)
            if p is not None:
                return p.device
        return torch.device("cpu")
    