import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class IDecoderHead(nn.Module, ABC):
    def __init__(self, transformer_hidden_dim, depth=2):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([
                nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
                nn.LayerNorm(transformer_hidden_dim),
                nn.ReLU(inplace=True)
            ])
        self.backbone = nn.Sequential(*layers)

    @abstractmethod
    def forward(self, feat):
        pass

class ActionDecoder(IDecoderHead):
    def __init__(self, transformer_hidden_dim, action_dims, depth=2):
        super().__init__(transformer_hidden_dim, depth)
        self.heads = nn.ModuleList([
            nn.Linear(transformer_hidden_dim, dim) for dim in action_dims
        ])

    def forward(self, feat):
        feat = self.backbone(feat)
        # actions = torch.cat([head(feat) for head in self.heads], dim=-1)
        return [head(feat) for head in self.heads]

class RewardDecoder(IDecoderHead):
    def __init__(self, transformer_hidden_dim, num_classes, depth=2):
        super().__init__(transformer_hidden_dim, depth)
        self.head = nn.Linear(transformer_hidden_dim, num_classes)

    def forward(self, feat):
        feat = self.backbone(feat)
        return self.head(feat)

class TerminationDecoder(IDecoderHead):
    def __init__(self, transformer_hidden_dim, depth=2):
        super().__init__(transformer_hidden_dim, depth)
        self.head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 1),
            # nn.Sigmoid() 
        )

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        return termination.squeeze(-1)