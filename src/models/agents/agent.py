import torch
import torch.nn as nn
import copy

from einops import rearrange
from src.models.attentive_pooler import AttentivePooler
from src.models.utils.losses import SymLogTwoHotLoss, EMAScalar
from src.models.agents.multidiscrete_actor import MultiCategoricalActor

def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage*len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per

def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=torch.float32):
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


class ActorCriticAgent(nn.Module):
    def __init__(self, feat_len, feat_dim, num_layers, hidden_dim, action_dim, gamma, lambd, entropy_coef) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)
        self.feat_len = feat_len

        #  Pooler
        self.attentive_pooler = AttentivePooler(
            num_queries=1,
            embed_dim=feat_dim,
            num_heads=12,
            mlp_ratio=4.0,
            depth=2,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            qkv_bias=True,
            complete_block=True,
        )

        # Actor network
        actor = [
            nn.Linear(feat_len*feat_dim, hidden_dim, bias=False),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(inplace=True)
        ]
        for i in range(num_layers - 1):
            actor.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.RMSNorm(hidden_dim),
                nn.SiLU(inplace=True)
            ])

        self.actor = MultiCategoricalActor(       
            preprocess_net=nn.Sequential(*actor),
            preprocess_net_dim=hidden_dim,
            action_dim=action_dim,
            )
        self.dist_fn = self.actor.dist_fn

        # Critic network
        critic = [
            nn.Linear(feat_len*feat_dim, hidden_dim, bias=False),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(inplace=True)
        ]
        for i in range(num_layers - 1):
            critic.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.RMSNorm(hidden_dim),
                nn.SiLU(inplace=True)
            ])

        self.critic = nn.Sequential(
            *critic,
            nn.Linear(hidden_dim, 255)
        )
        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, eps=1e-5)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))

    def policy(self, x):
        logits = self.actor(x)
        return logits

    def value(self, x):
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def slow_value(self, x):
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def get_logits_raw_value(self, x):
        logits = self.actor(x)
        raw_value = self.critic(x)
        return logits, raw_value

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        self.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            B, T, P, D = latent.shape
            latent = rearrange(latent,"B T P D -> (B T) P D")
            latent = self.attentive_pooler(latent).squeeze(1)
            latent = rearrange(latent,"(B T) D -> B (T D)", T=T)
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
    
    def pool_sliding_window(self, latent, clip_len):
        """
        latent: [B, T_full, P, D]
        clip_len: T
        return: [B, num_clips, D]
        """
        B, T_full, P, D = latent.shape
        num_clips = T_full - clip_len + 1
        assert num_clips > 0, "clip_len too long, can't do sliding window"

        latent = rearrange(latent,"B T P D -> (B T) P D")
        latent = self.attentive_pooler(latent).squeeze(1)
        latent = rearrange(latent,"(B T) D -> B T D", T=T_full)

        windows = [rearrange(latent[:, t:t+clip_len], "B T D -> B 1 (T D)") for t in range(num_clips)]  
        latent = torch.cat(windows, dim=1)  
        return latent
    
    def update(self, latent, action, old_logprob, old_value, reward, termination, clip_len, logger=None):
        '''
        Update policy and value model
        '''
        self.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # latent.shape:torch.Size([64, 17, 1536])
            # logits.shape:torch.Size([64, 17, 15])
            # action:torch.Size([64, 16, 2])
            # log_prob.shape:torch.Size([64, 16])
            latent = self.pool_sliding_window(latent, self.feat_len)
            logits, raw_value = self.get_logits_raw_value(latent)
            # dist = distributions.Categorical(logits=logits[:, :-1])
            dist = self.dist_fn(logits[:, :-1,:])
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # decode value, calc lambda return
            slow_value = self.slow_value(latent)
            slow_lambda_return = calc_lambda_return(reward, slow_value, termination, self.gamma, self.lambd)
            value = self.symlog_twohot_loss.decode(raw_value)
            lambda_return = calc_lambda_return(reward, value, termination, self.gamma, self.lambd)

            # update value function with slow critic regularization
            value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach())
            slow_value_regularization_loss = self.symlog_twohot_loss(raw_value[:, :-1], slow_lambda_return.detach())

            lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound-lower_bound
            norm_ratio = torch.max(torch.ones(1).cuda(), S)  # max(1, S) in the paper
            norm_advantage = (lambda_return-value[:, :-1]) / norm_ratio
            policy_loss = -(log_prob * norm_advantage.detach()).mean()

            entropy_loss = entropy.mean()

            loss = policy_loss + value_loss + slow_value_regularization_loss - self.entropy_coef * entropy_loss

        # gradient descent
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.update_slow_critic()

        if logger is not None:
            logger.log('ActorCritic/policy_loss', policy_loss.item())
            logger.log('ActorCritic/value_loss', value_loss.item())
            logger.log('ActorCritic/entropy_loss', entropy_loss.item())
            logger.log('ActorCritic/S', S.item())
            logger.log('ActorCritic/norm_ratio', norm_ratio.item())
            logger.log('ActorCritic/total_loss', loss.item())
