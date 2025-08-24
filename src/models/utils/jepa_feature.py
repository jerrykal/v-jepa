from __future__ import annotations
from dataclasses import dataclass
from torch import Tensor


@dataclass(frozen=True)
class TPShape:
    T: int
    P: int


class StateFeature:
    """
    FeatureTensor: A lightweight container to maintain [B, T, P, D] / [B, T*P, D] tensors.

    - Internally always stored as [B, T, P, D].
    - Provides convenient views into [B, T*P, D].
    - Keeps metadata (T, P) so you don't need to pass them around manually.
    - Supports common slicing, padding, concatenation, and device/dtype operations.
    """

    __slots__ = ("_x", "_shape")

    def __init__(self, x: Tensor, t:int, p:int):
        assert x.ndim == 3, f"expected [B,(T*P),D], got {tuple(x.shape)}"
        B, TP, D = x.shape
        assert t*p == TP, f"shape/meta mismatch: tensor {(TP)} vs meta {(t, p)}"
        self._x = x
        self._tp_shape = TPShape(T=t, P=p)

    # >> Constructors 
    @classmethod
    def from_time_patches(cls, x: Tensor) -> StateFeature:
        """
        Create from a [B, T P, D] tensor.
        """
        assert x.ndim == 4, f"expected [B, T, P,D], got {tuple(x.shape)}"
        B, T, P, D = x.shape

        return cls(x.contiguous().view(B, T*P, D), t=T, p=P)
    
    # >> Views / Accessors
    @property
    def x(self) -> Tensor:
        """Return internal tensor [B, T, P, D]."""
        return self._x

    @property
    def B(self) -> int: return self._x.shape[0]
    @property
    def T(self) -> int: return self._tp_shape.T
    @property
    def P(self) -> int: return self._tp_shape.P
    @property
    def D(self) -> int: return self._x.shape[-1]
    @property
    def device(self):   return self._x.device
    @property
    def dtype(self):    return self._x.dtype

    def as_time_patches(self) -> Tensor:
        """Return as [B, T, P, D]."""
        return self._x.contiguous().view(self.B, self.T, self.P, self.D)

    def as_flat(self) -> Tensor:
        """Return as [B, T*P, D]."""
        return self._x
    
    def detach(self) -> StateFeature:
        """Return a detached copy."""
        return StateFeature(self._x.detach(), t=self._tp_shape.T, P=self._tp_shape.P)