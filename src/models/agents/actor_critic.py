# actor_critic.py
import torch
import torch.nn as nn
from typing import List

from src.models.agents.multidiscrete_actor import MultiCategoricalActor

def build_mlp(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    output_dim: int | None = None,
    activation: nn.Module = nn.SiLU(inplace=True),
    norm_layer: nn.Module = nn.RMSNorm
) -> nn.Sequential:
    """
    Build a simple feed-forward MLP:
    [Linear -> Norm -> Activation] * num_layers
    Optionally add an output layer at the end.
    """
    layers = [
        nn.Linear(input_dim, hidden_dim, bias=False),
        norm_layer(hidden_dim),
        activation
    ]
    for _ in range(num_layers - 1):
        layers += [
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            norm_layer(hidden_dim),
            activation
        ]
    if output_dim is not None:
        layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Actor network for multi-discrete action spaces.

    Input:
        - x: tensor of shape [B, F] or [B, N, F]
          (B = batch, N = number of windows/clips, F = feature dimension)
    Output:
        - logits: raw action logits (structure depends on MultiCategoricalActor)

    Notes:
        - Uses an internal MLP for feature preprocessing
        - Then delegates to MultiCategoricalActor to produce action logits
        - Provides a .dist() helper to construct the corresponding distribution
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        action_dim: List[int],
        activation: nn.Module = nn.SiLU(inplace=True),
        norm_layer: nn.Module = nn.RMSNorm,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Multi-discrete actor head
        # We pass Identity() here because preprocessing is already done
        self.core = MultiCategoricalActor(
            preprocess_net=build_mlp(
                input_dim, hidden_dim, num_layers,
                activation=activation, norm_layer=norm_layer
            ),
            preprocess_net_dim=hidden_dim,
            action_dim=action_dim,
        )

        # Expose the distribution factory from MultiCategoricalActor
        self.dist_fn = self.core.dist_fn

    def _forward_2d(self, x2d: torch.Tensor):
        """
        Forward pass for 2D input [B, F].
        """
        h = self.preprocess(x2d)   # [B, H]
        logits = self.core(h)      # Action logits
        return logits

    def forward(self, x: torch.Tensor):
        """
        Forward pass that supports both [B, F] and [B, N, F].
        If 3D input is given, it is reshaped to 2D, processed,
        and reshaped back to [B, N, ...].
        """
        if x.dim() == 2:
            return self._forward_2d(x)
        elif x.dim() == 3:
            B, N, F = x.shape
            x2d = x.reshape(B * N, F)
            out2d = self._forward_2d(x2d)
            # If the output is a tuple/list of tensors, reshape each
            if isinstance(out2d, (tuple, list)):
                return type(out2d)(o.reshape(B, N, *o.shape[1:]) for o in out2d)
            else:
                return out2d.reshape(B, N, *out2d.shape[1:])
        else:
            raise ValueError(f"Actor expects [B,F] or [B,N,F], got {tuple(x.shape)}")

    def dist(self, logits):
        """
        Convenience wrapper to build the action distribution
        from raw logits using MultiCategoricalActor's dist_fn.
        """
        return self.dist_fn(logits)


class Critic(nn.Module):
    """
    Critic network that predicts value distributions (Two-Hot encoding).

    Input:
        - x: tensor of shape [B, F] or [B, N, F]

    Output:
        - raw_value: tensor of shape [B, output_bins] or [B, N, output_bins]
                     (logits before decoding)

    Notes:
        - Typically, SymLogTwoHotLoss.decode() will be used
          externally to convert logits into scalar values.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_bins: int = 255,
        activation: nn.Module = nn.SiLU(inplace=True),
        norm_layer: nn.Module = nn.RMSNorm,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_bins = output_bins

        # MLP outputs logits for value distribution
        self.mlp = build_mlp(
            input_dim, hidden_dim, num_layers,
            output_dim=output_bins,
            activation=activation, norm_layer=norm_layer
        )

    def _forward_2d(self, x2d: torch.Tensor):
        """
        Forward pass for 2D input [B, F].
        """
        return self.mlp(x2d)  # [B, output_bins]

    def forward(self, x: torch.Tensor):
        """
        Forward pass that supports both [B, F] and [B, N, F].
        """
        if x.dim() == 2:
            return self._forward_2d(x)
        elif x.dim() == 3:
            B, N, F = x.shape
            x2d = x.reshape(B * N, F)
            y2d = self._forward_2d(x2d)
            return y2d.reshape(B, N, self.output_bins)
        else:
            raise ValueError(f"Critic expects [B,F] or [B,N,F], got {tuple(x.shape)}")
