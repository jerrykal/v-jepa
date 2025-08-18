import torch
import torch.nn.functional as F

class ActionParser:
    '''
    ActionParser handles conversion between structured raw actions and one-hot tensors.

    Example:
        Raw action:
            dict version:
            - {
                "movement": "up",
                "combat": "attack"
            }
            list version:
            - [4, 3]


        One-hot tensor:
            [0, 0, 1, 0,   # movement: up 
             1, 0, 0]      # combat: attack

    Features:
    - Converts multi-class discrete actions to concatenated one-hot tensor.
    - Supports reverse mapping from tensor back to raw action.
    - Designed for global/shared usage — should be initialized once after action configuration is known.
    '''
    category_list = []
    action_dict = {}
    value_to_index = {}
    index_to_value = {}
    action_dims = []
    total_dim = 0
    _initialized = False
    _use_named_keys = True

    @staticmethod
    def init(config):
        '''
        Initialize with either a dict or a list.

        Args:
            config: dict (named categories) or list (anonymous dimensions)
                - dict: {"move": ["up", "down"], "attack": ["hit", "block"]}
                - list: [3, 4]  → two categories with 3 and 4 possible actions
        '''
        if isinstance(config, dict):
            ActionParser._use_named_keys = True
            ActionParser.category_list = list(config.keys())
            ActionParser.action_dict = config
            ActionParser.value_to_index = {
                cat: {v: i for i, v in enumerate(vals)}
                for cat, vals in config.items()
            }
            ActionParser.index_to_value = {
                cat: {i: v for i, v in enumerate(vals)}
                for cat, vals in config.items()
            }
            ActionParser.action_dims = [len(vals) for vals in config.values()]
        elif isinstance(config, list):
            ActionParser._use_named_keys = False
            ActionParser.category_list = [f"cat{i}" for i in range(len(config))]
            ActionParser.action_dict = {
                name: list(range(dim)) for name, dim in zip(ActionParser.category_list, config)
            }
            ActionParser.value_to_index = {
                name: {v: i for i, v in enumerate(vals)}
                for name, vals in ActionParser.action_dict.items()
            }
            ActionParser.index_to_value = {
                name: {i: v for i, v in enumerate(vals)}
                for name, vals in ActionParser.action_dict.items()
            }
            ActionParser.action_dims = config
        else:
            raise ValueError("ActionParser.init only accepts dict or list input.")

        ActionParser.total_dim = sum(ActionParser.action_dims)
        ActionParser._initialized = True

    @staticmethod
    def encode(raw_action: list | dict | torch.Tensor) -> torch.Tensor:
        ActionParser._check_initialized()

        if ActionParser._use_named_keys:
            # 仍只支持单样本 dict（如需 dict 的批/序列，后续可按同思路扩展）
            assert isinstance(raw_action, dict), "Expected dict input."
            one_hots = []
            for cat in ActionParser.category_list:
                idx = ActionParser.value_to_index[cat][raw_action[cat]]
                one_hot = F.one_hot(torch.tensor(idx), num_classes=len(ActionParser.action_dict[cat]))
                one_hots.append(one_hot)
            return torch.cat(one_hots).float()

        dims = ActionParser.action_dims
        C = len(dims)

        if isinstance(raw_action, list):
            ra = torch.tensor(raw_action, dtype=torch.long)
        elif isinstance(raw_action, torch.Tensor):
            ra = raw_action.to(dtype=torch.long)
        else:
            raise TypeError("In list mode, raw_action must be list or torch.Tensor.")

        if ra.ndim == 1:
            # [C]
            assert ra.numel() == C, f"Expected {C} categories, got {ra.numel()}."
            parts = [F.one_hot(ra[i], num_classes=dims[i]) for i in range(C)]
            return torch.cat(parts, dim=0).float()  # [total_dim]
        elif ra.ndim == 2:
            # [B, C]
            B, C_in = ra.shape
            assert C_in == C, f"Expected {C} categories, got {C_in}."
            parts = [F.one_hot(ra[:, i], num_classes=dims[i]) for i in range(C)]  # [B, dim_i]
            return torch.cat(parts, dim=1).float()  # [B, total_dim]
        elif ra.ndim == 3:
            B, L, C_in = ra.shape
            assert C_in == C, f"Expected {C} categories, got {C_in}."
            ra2 = ra.reshape(B * L, C) 
            parts = [F.one_hot(ra2[:, i], num_classes=dims[i]) for i in range(C)]  # [B*L, dim_i]
            out = torch.cat(parts, dim=1).float()  # [B*L, total_dim]
            return out.reshape(B, L, -1)  # [B, L, total_dim]

        else:
            raise ValueError(f"Unsupported raw_action.ndim={ra.ndim}; expected 1, 2 or 3 in list mode.")

    @staticmethod
    def decode(tensor: torch.Tensor) -> list | dict:
        ActionParser._check_initialized()

        dims = ActionParser.action_dims
        total = sum(dims)

        if ActionParser._use_named_keys:
            assert tensor.ndim == 1 and tensor.numel() == total, \
                f"Expected 1D tensor of length {total} for dict mode."
            result = {}
            start = 0
            for cat, dim in zip(ActionParser.category_list, dims):
                sub = tensor[start:start+dim]
                idx = int(torch.argmax(sub))
                result[cat] = ActionParser.index_to_value[cat][idx]
                start += dim
            return result

        if tensor.ndim == 1:
            assert tensor.numel() == total, f"Expected total_dim={total}, got {tensor.numel()}."
            out = []
            start = 0
            for cat, dim in zip(ActionParser.category_list, dims):
                sub = tensor[start:start+dim]
                idx = int(torch.argmax(sub))
                out.append(idx)
                start += dim
            return out  
        elif tensor.ndim == 2:
            # [B, total]
            B, T = tensor.shape
            assert T == total, f"Expected total_dim={total}, got {T}."
            idx_parts = []
            start = 0
            for dim in dims:
                sub = tensor[:, start:start+dim]         # [B, dim]
                idx = torch.argmax(sub, dim=-1)          # [B]
                idx_parts.append(idx)
                start += dim
            return torch.stack(idx_parts, dim=-1).tolist()  
        elif tensor.ndim == 3:
            B, L, T = tensor.shape
            assert T == total, f"Expected total_dim={total}, got {T}."
            idx_parts = []
            start = 0
            for dim in dims:
                sub = tensor[..., start:start+dim]        # [B, L, dim]
                idx = torch.argmax(sub, dim=-1)           # [B, L]
                idx_parts.append(idx)
                start += dim
            return torch.stack(idx_parts, dim=-1).tolist()

        else:
            raise ValueError("decode expects tensor of shape [total], [B, total], or [B, L, total].")
        
    @staticmethod
    def _check_initialized():
        if not ActionParser._initialized:
            raise RuntimeError("ActionParser not initialized. Call `ActionParser.init(...)` first.")
        

