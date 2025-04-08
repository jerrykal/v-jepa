import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from einops import reduce



@torch.no_grad()
def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.no_grad()
def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class SymLogLoss(nn.Module):
    def __init__(self,reduce="sum,mean"):
        super().__init__()
        self._loss = MSELoss(reduce)

    def forward(self, output, target):
        return 0.5 * self._loss(output, target)

    
class SymLogTwoHotLoss(nn.Module):
    def __init__(self, num_classes, lower_bound, upper_bound):
        super().__init__()
        self.num_classes = num_classes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bin_length = (upper_bound - lower_bound) / (num_classes-1)

        # use register buffer so that bins move with .cuda() automatically
        self.bins: torch.Tensor
        self.register_buffer(
            'bins', torch.linspace(-20, 20, num_classes), persistent=False)

    def forward(self, output, target):
        target = symlog(target)
        assert target.min() >= self.lower_bound and target.max() <= self.upper_bound

        index = torch.bucketize(target, self.bins)
        diff = target - self.bins[index-1]  # -1 to get the lower bound
        weight = diff / self.bin_length
        weight = torch.clamp(weight, 0, 1)
        weight = weight.unsqueeze(-1)

        target_prob = (1-weight)*F.one_hot(index-1, self.num_classes) + weight*F.one_hot(index, self.num_classes)
        loss = -target_prob * F.log_softmax(output, dim=-1)
        loss = loss.sum(dim=-1)
        return loss.mean()

    def decode(self, output):
        return symexp(F.softmax(output, dim=-1) @ self.bins)
    
class MSELoss(nn.Module):
    def __init__(self, reduce="sum,mean") -> None:
        super().__init__()
        reduce_func_list = reduce.split(",")
        self.feat_reduce = reduce_func_list[0]
        self.batch_reduce = reduce_func_list[1]


    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L C H W -> B L", self.feat_reduce)
        loss = reduce(loss, "B L -> ", self.batch_reduce)
        return loss
    
class GeneralizedLpLoss(nn.Module):
    def __init__(self, p=1.0, reduce="mean,mean"):
        super().__init__()
        assert p > 0, "p must be positive"
        self.p = p
        reduce_func_list = reduce.split(",")
        self.feat_reduce = reduce_func_list[0]
        self.batch_reduce = reduce_func_list[1]

    def forward(self, pred, target):
        # pred, target shape: [B, N, D] or anything
        loss = torch.abs(pred - target) ** self.p  # element-wise |x|^p
        loss = reduce(loss, "B N D -> B ", self.feat_reduce)
        loss = reduce(loss, "B -> ", self.batch_reduce)
        return loss / self.p
    
class EMAScalar():
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar


if __name__ == "__main__":
    loss_func = SymLogTwoHotLoss(255, -20, 20)
    output = torch.randn(1, 1, 255).requires_grad_()
    target = torch.ones(1).reshape(1, 1).float() * 0.1
    print(target)
    loss = loss_func(output, target)
    print(loss)

    # prob = torch.ones(1, 1, 255)*0.5/255
    # prob[0, 0, 128] = 0.5
    # logits = torch.log(prob)
    # print(loss_func.decode(logits), loss_func.bins[128])
