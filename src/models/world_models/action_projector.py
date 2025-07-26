import torch
import torch.nn as nn
from src.utils.action_parser import ActionParser

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
    def __init__(self):
        super().__init__()
        # You can define a simple MLP here based on ActionParser.total_dim
        pass

    def forward(self, x):
        # Implement the forward projection logic here
        pass
