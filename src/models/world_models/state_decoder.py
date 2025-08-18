import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from src.models.attentive_pooler import AttentivePooler

class IDecoderHead(nn.Module, ABC):
    def __init__(self, input_dim, hidden_dim, depth=2):
        super().__init__()
        layers = []
        _input_dim = input_dim
        for _ in range(depth):
            layers.extend([
                nn.Linear(_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            _input_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

    @abstractmethod
    def forward(self, feat):
        pass

class RewardsDecoder(IDecoderHead):
    '''
    RewardsDecoder:
    A simple neural network module intended to decode predicted latent features into scalar reward values.

    Usage:
    - Input: latent feature tensor (e.g., from world model)
    - Output: predicted reward value(s)
    '''
    def __init__(self,
                 num_classes, input_dim, hidden_dim, depth=2 ):
        super().__init__(input_dim, hidden_dim, depth)
        self.proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, pooler, feat):
        x = pooler(feat).squeeze(1) # [B N D] -> [B Q D]
        x = self.backbone(x) 
        x = self.proj(x)
        return x


class TerminationDecoder(IDecoderHead):
    '''
    TerminationDecoder:
    A neural network module to predict whether a trajectory should terminate,
    based on latent features.

    Usage:
    - Input: latent feature tensor
    - Output: termination flag (e.g., binary 0/1 or probability)
    '''
    def __init__(self, input_dim, hidden_dim, depth=2 ):
        super().__init__(input_dim, hidden_dim, depth)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, pooler, feat):
        x = pooler(feat).squeeze(1) # [B N D] -> [B Q D]
        x = self.backbone(x) 
        x = self.proj(x).squeeze(1)
        return x
