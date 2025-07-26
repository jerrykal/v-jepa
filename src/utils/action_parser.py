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
    def encode(raw_action: list | dict) -> torch.Tensor:
        '''
        Encode raw action into one-hot tensor.

        Args:
            raw_action: list or dict, format depends on init mode

        Returns:
            1D tensor [total_dim]
        '''
        ActionParser._check_initialized()

        if ActionParser._use_named_keys:
            assert isinstance(raw_action, dict), "Expected dict input."
            one_hots = []
            for cat in ActionParser.category_list:
                idx = ActionParser.value_to_index[cat][raw_action[cat]]
                one_hot = F.one_hot(torch.tensor(idx), num_classes=len(ActionParser.action_dict[cat]))
                one_hots.append(one_hot)
        else:
            assert isinstance(raw_action, list), "Expected list input."
            one_hots = []
            for i, idx in enumerate(raw_action):
                dim = ActionParser.action_dims[i]
                one_hot = F.one_hot(torch.tensor(idx), num_classes=dim)
                one_hots.append(one_hot)

        return torch.cat(one_hots).float()

    @staticmethod
    def decode(tensor: torch.Tensor) -> list | dict:
        '''
        Decode one-hot tensor back to structured raw action.

        Args:
            tensor: 1D tensor [total_dim]

        Returns:
            list or dict, depending on init mode
        '''
        ActionParser._check_initialized()

        result = {} if ActionParser._use_named_keys else []
        start = 0
        for cat, dim in zip(ActionParser.category_list, ActionParser.action_dims):
            sub = tensor[start:start+dim]
            idx = torch.argmax(sub).item()
            val = ActionParser.index_to_value[cat][idx]
            if ActionParser._use_named_keys:
                result[cat] = val
            else:
                result.append(idx)
            start += dim
        return result

    @staticmethod
    def _check_initialized():
        if not ActionParser._initialized:
            raise RuntimeError("ActionParser not initialized. Call `ActionParser.init(...)` first.")