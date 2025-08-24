import torch.nn as nn

_NORM_MAP = {
    "batchnorm1d": nn.BatchNorm1d,
    "batchnorm2d": nn.BatchNorm2d,
    "batchnorm3d": nn.BatchNorm3d,
    "layernorm": nn.LayerNorm,
    "instancenorm1d": nn.InstanceNorm1d,
    "instancenorm2d": nn.InstanceNorm2d,
    "instancenorm3d": nn.InstanceNorm3d,
    "groupnorm": nn.GroupNorm,
    "identity": nn.Identity,
}

def get_norm_layer(norm_type: str):
    norm_type = norm_type.lower()
    if norm_type not in _NORM_MAP:
        raise ValueError(f"Unsupported norm layer type: {norm_type}")
    return _NORM_MAP[norm_type]