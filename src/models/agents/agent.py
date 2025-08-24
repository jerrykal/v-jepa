import torch
import torch.nn as nn
import copy
from einops import rearrange
from typing import List
from torch.optim import Optimizer
from src.utils.logging import TensorboardLogger
from src.utils.logging import get_logger, grad_logger, adamw_logger
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule
from src.models.attentive_pooler import AttentivePooler
from src.models.utils.losses import SymLogTwoHotLoss, EMAScalar
from src.models.agents.multidiscrete_actor import MultiCategoricalActor
from src.models.utils.jepa_feature import StateFeature
from src.models.agents.actor_critic import Actor, Critic

logger = get_logger(__name__)

def pool_sliding_window(
        pooler:AttentivePooler, 
        latent:torch.Tensor, 
        clip_len:int):
    """
    latent: [B, T_full, P, D]
    clip_len: T
    return: [B, num_clips, D]
    """
    B, T_full, P, D = latent.shape
    num_clips = T_full - clip_len + 1
    assert num_clips > 0, "clip_len too long, can't do sliding window"

    latent = rearrange(latent,"B T P D -> (B T) P D")
    latent = pooler(latent).squeeze(1)
    latent = rearrange(latent,"(B T) D -> B T D", T=T_full)

    windows = [rearrange(latent[:, t:t+clip_len], "B T D -> B 1 (T D)") for t in range(num_clips)]  
    latent = torch.cat(windows, dim=1)  
    return latent

def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage*len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per

def lambda_return_fn(rewards, values, termination, gamma, lam, dtype=torch.float32):
    # Invert termination to have 0 if the episode ended and 1 otherwise
    inv_termination = (termination * -1) + 1

    batch_size, batch_length = rewards.shape[:2]
    # gae_step = torch.zeros((batch_size, ), dtype=dtype, device="cuda")
    gamma_return = torch.zeros((batch_size, batch_length+1), dtype=dtype, device="cuda")
    gamma_return[:, -1] = values[:, -1]
    for t in reversed(range(batch_length)):  # with last bootstrap
        gamma_return[:, t] = \
            rewards[:, t] + \
            gamma * inv_termination[:, t] * (1-lam) * values[:, t] + \
            gamma * inv_termination[:, t] * lam * gamma_return[:, t+1]
    return gamma_return[:, :-1]

class ActorCriticAgent():
    def __init__(self, 
                # Models
                 actor:Actor,
                 critic:Critic,
                 pooler:AttentivePooler,

                # Optimizer
                 optimizer:Optimizer, 
                 scaler:torch.amp.GradScaler, 
                 lr_scheduler:WarmupCosineSchedule, 
                 wd_scheduler:CosineWDSchedule,

                # Debug
                 tb_logger:TensorboardLogger | None,

                # Training setting
                 feat_len:int,
                 gamma:float, lambd:float, entropy_coef:float,
                 use_amp:bool,
                 amp_dtype:torch.dtype,
                 warmup:int|None=None,
                 clip_grad:float=10.0,
            ) -> None:
        super().__init__()
        # >> Models setting
        self._pooler = pooler

        self._actor = actor
        self.dist_fn = self._actor.dist_fn

        self._critic = critic
        self._slow_critic = copy.deepcopy(self._critic)
        self.modules: List[nn.Module] = [self._pooler, self._actor, self._critic, self._slow_critic]


        # >> EMA scalars 
        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        # >> Loss
        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)

        # >> Optimization 
        self._optimizer = optimizer
        self._scaler = scaler
        self._lr_scheduler = lr_scheduler
        self._wd_scheduler = wd_scheduler

        # >> Debug setting
        self._tb_logger = tb_logger

        # >> Training Setting
        self.feat_len = feat_len
        self.gamma = gamma
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        self._warmup = warmup if warmup is not None else self._lr_scheduler.warmup_steps
        self._clip_grad = clip_grad
        self._use_amp = use_amp
        self._amp_dtype = amp_dtype
        self._step = 0

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(self._slow_critic.parameters(), self._critic.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))

    @torch.no_grad()
    def slow_value(self, x):
        value = self._slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value
    
    def policy(self, x):
        logits = self._actor(x)
        return logits

    def value(self, x):
        value = self._critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def get_logits_raw_value(self, x):
        logits = self._actor(x)
        raw_value = self._critic(x)
        return logits, raw_value
    
    def train(self):
        for m in self.modules:
            m.train()
            
    def eval(self):
        for m in self.modules:
            m.eval()

    @torch.no_grad()
    def sample(self, feature:StateFeature, greedy=False):
        self.eval()
        with torch.amp.autocast(device_type=feature.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
            latent = feature.as_time_patches() # [B T P D]
            latent = pool_sliding_window(latent, self.feat_len)[:,-1,:,:]
            logits = self.policy(latent)
            dist = self.dist_fn(logits)
            if greedy:
                action = dist.mode()
            else:
                action = dist.sample()
        return action

    def sample_as_env_action(self, latent, greedy=False):
        action = self.sample(latent, greedy)
        return action.detach().cpu().squeeze(-1).numpy()
    
    def update(self, feature:StateFeature, action, old_logprob, old_value, reward, termination, clip_len, logger=None):
        self.train()
        # step LR and WD schedulers
        _new_lr = self._lr_scheduler.step()
        _new_wd = self._wd_scheduler.step()
        
        with torch.amp.autocast(device_type=feature.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
            latent = feature.as_time_patches()
            latent = pool_sliding_window(latent, self.feat_len) 
            logits, raw_value = self.get_logits_raw_value(latent)
            dist = self.dist_fn(logits[:, :-1,:])
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            entropy_loss = self.entropy_coef * entropy.mean()

            # decode value, calc lambda return
            slow_value = self.slow_value(latent)
            slow_lambda_return = lambda_return_fn(reward, slow_value, termination, self.gamma, self.lambd)
            value = self.symlog_twohot_loss.decode(raw_value)
            lambda_return = lambda_return_fn(reward, value, termination, self.gamma, self.lambd)

            # update value function with slow critic regularization
            value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach())
            slow_value_regularization_loss = self.symlog_twohot_loss(raw_value[:, :-1], slow_lambda_return.detach())

            lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound-lower_bound
            norm_ratio = torch.max(torch.ones(1).cuda(), S)  # max(1, S) in the paper
            norm_advantage = (lambda_return-value[:, :-1]) / norm_ratio
            policy_loss = -(log_prob * norm_advantage.detach()).mean()

            loss = policy_loss + value_loss + slow_value_regularization_loss - entropy_loss

        # gradient descent
        if self._use_amp:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)  # for clip grad
        else:
            loss.backward()

        if (self._step > self._warmup) and (self._clip_grad is not None):
            _pooler_norm = torch.nn.utils.clip_grad_norm_(self._pooler.parameters(), self._clip_grad)
            _actor_norm = torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._clip_grad)
            _critic_norm = torch.nn.utils.clip_grad_norm_(self._critic.parameters(), self._clip_grad)

        if self._use_amp:
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            self._optimizer.step()

        grad_stats_pooler = grad_logger(self._pooler.named_parameters())
        grad_stats_pooler.global_norm = float(_pooler_norm)
        grad_stats_actor = grad_logger(self._actor.named_parameters())
        grad_stats_actor.global_norm = float(_actor_norm)
        grad_stats_critic = grad_logger(self._critic.named_parameters())
        grad_stats_critic.global_norm = float(_critic_norm)

        self._optimizer.zero_grad(set_to_none=True)
        optim_stats = adamw_logger(self._optimizer)

        self.update_slow_critic()
        self._step += 1
        if self._tb_logger is not None:
            self._tb_logger.log('ActorCritic/policy_loss', policy_loss.item())
            self._tb_logger.log('ActorCritic/value_loss', value_loss.item())
            self._tb_logger.log('ActorCritic/entropy_loss', entropy_loss.item())
            self._tb_logger.log('ActorCritic/S', S.item())
            self._tb_logger.log('ActorCritic/norm_ratio', norm_ratio.item())
            self._tb_logger.log('ActorCritic/total_loss', loss.item())

        return {
            "loss":{
                "total":loss.item(),
                "policy":policy_loss.item(),
                "value":value_loss.item(),
                "entropy":entropy_loss.item()
            },
            "train_model_state":{
                "pooler":grad_stats_pooler,
                "actor":grad_stats_actor,
                "critic":grad_stats_critic,
            },
            "optim_state": optim_stats,
            "lr":_new_lr,
            "wd":_new_wd,
        }