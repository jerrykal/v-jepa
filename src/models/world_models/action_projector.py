import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.action_parser import ActionParser
from src.models.utils.quantization import VectorQuantization


class ActionProjector(nn.Module):
    '''
    ActionProjector:
    A neural network module that maps real (environment-level) actions into the pseudo-action space.

    Purpose:
    - To bridge the gap between real-world agent actions and the latent action representations.
    - Enables the use of real actions at inference time by projecting them into the learned latent space.

    Usage:
    - Input: Raw real action (e.g, [3, 4])
    - Output: pseudo-action vector (same space as latent action encoder output)
    '''
    def __init__(self, hidden_dims, depth, quant:VectorQuantization):
        super().__init__()
        self._qunt = quant
        input_dim = ActionParser.total_dim  # Assume static variable
        
        layers = []

        # First layer: input → hidden
        layers.append(nn.Linear(input_dim, hidden_dims))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers: hidden → hidden (depth - 2)
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.ReLU(inplace=True))

        # Final layer: hidden → output
        layers.append(nn.Linear(hidden_dims, quant._embedding_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, act):
        x = ActionParser.encode(act)
        x = self.model(x)
        with torch.no_grad():
            quantized, idxs = self._qunt.quantize(x)
            outs = self._qunt.out_proj(quantized)
        return (outs, idxs), x
