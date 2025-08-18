from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class Categorical(torch.distributions.Categorical):
    """
    Mostly interface changes, add mode() function, no real difference from Categorical
    """

    def mode(self):
        return self.logits.argmax(dim=-1)

    def imitation_loss(self, actions, reduction="mean"):
        """
        actions: groundtruth actions from expert
        """
        assert actions.dtype == torch.long
        if self.logits.ndim == 3:
            assert actions.ndim == 2
            assert self.logits.shape[:2] == actions.shape
            return F.cross_entropy(
                self.logits.reshape(-1, self.logits.shape[-1]),
                actions.reshape(-1),
                reduction=reduction,
            )
        return F.cross_entropy(self.logits, actions, reduction=reduction)

    def random_actions(self):
        """
        Generate a completely random action, NOT the same as sample(), more like
        action_space.sample()
        """
        return torch.randint(
            low=0,
            high=self.logits.size(-1),
            size=self.logits.size()[:-1],
            device=self.logits.device,
        )


class MultiCategorical(torch.distributions.Distribution):
    def __init__(self, logits, action_dims: list[int]):
        # assert logits.dim() == 2, logits.shape
        super().__init__(batch_shape=logits[:1], validate_args=False)
        self._action_dims = tuple(action_dims)
        assert logits.size(-1) == sum(
            self._action_dims
        ), f"sum of action dims {self._action_dims} != {logits.size(1)}"
        self._dists = [
            Categorical(logits=split)
            for split in torch.split(logits, action_dims, dim=-1)
        ]

    def log_prob(self, actions):
        return torch.stack(
            [
                dist.log_prob(action)
                for dist, action in zip(self._dists, torch.unbind(actions, dim=-1))
            ],
            dim=-1,
        ).sum(dim=-1)

    def entropy(self):
        return torch.stack([dist.entropy() for dist in self._dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        assert sample_shape == torch.Size()
        return torch.stack([dist.sample() for dist in self._dists], dim=-1)

    def mode(self):
        # return torch.stack(
        #     [torch.argmax(dist.probs, dim=1) for dist in self._dists], dim=-1
        # )
        return torch.stack(
            [dist.mode() for dist in self._dists], dim=-1
        )

    def imitation_loss(self, actions, weights=None, reduction="mean"):
        """
        Args:
            actions: groundtruth actions from expert
            weights: weight the imitation loss from each component in MultiDiscrete
            reduction: "mean" or "none"

        Returns:
            one torch float
        """
        assert actions.dtype == torch.long
        assert actions.shape[-1] == len(self._action_dims)
        assert reduction in ["mean", "none"]
        if weights is None:
            weights = [1.0] * len(self._dists)
        else:
            assert len(weights) == len(self._dists)

        aggregate = sum if reduction == "mean" else list
        return aggregate(
            dist.imitation_loss(a, reduction=reduction) * w
            for dist, a, w in zip(self._dists, torch.unbind(actions, dim=-1), weights)
        )

    def random_actions(self):
        return torch.stack([dist.random_actions() for dist in self._dists], dim=-1)


class MultiCategoricalActor(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        *,
        preprocess_net_dim:int,
        action_dim: list[int],
        hidden_dim: int = 0,
        hidden_depth: int = 0,
        activation: str = "relu",
    ):
        super().__init__()
        self.mlps = nn.ModuleList()
        self.preprocess = preprocess_net
        for action in action_dim:
            net = nn.Sequential(
                nn.Linear(preprocess_net_dim, action),
                nn.ReLU(action)
            )
            self.mlps.append(net)
        self._action_dim = action_dim
    

    def forward(self, x, state=None, info=None):
        x = self.preprocess(x)
        return torch.cat([mlp(x) for mlp in self.mlps], dim=-1)

    @property
    def dist_fn(self):
        return lambda x: MultiCategorical(logits=x, action_dims=self._action_dim)
