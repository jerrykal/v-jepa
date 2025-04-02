import torch
import torch.nn as nn
from src.models.world_models.base_world_model import WorldModelBase
class JEPAWorldModel(WorldModelBase):
    def __init__(self):
        super().__init__()
        