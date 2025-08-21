import torch
import torch.nn as nn

from typing import List
from einops import rearrange
from torch.optim import Optimizer
from src.utils.logging import TensorboardLogger
from src.utils.logging import get_logger, grad_logger, adamw_logger
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule
from src.models.utils.losses import SymLogTwoHotLoss, MSELoss
from src.models.utils.jepa_feature import StateFeature
    
# Models
from src.models.attentive_pooler import AttentivePooler
from src.models.utils.multimask import (
    MultiMaskWrapper, PredictorMultiMaskWrapper, LatentActionEncoderMultiMaskWrapper)
from src.models.world_models.state_decoder import RewardsDecoder as RewardsDec
from src.models.world_models.state_decoder import TerminationDecoder as TerminDec 
from src.models.world_models.action_projector import ActionProjector 

logger = get_logger(__name__)

class WorldModel():
    '''
    The class is organize the each modules on training loop 
    '''
    def __init__(self,
                 # Models
                 context_encoder:MultiMaskWrapper,
                 target_encoder:MultiMaskWrapper,
                 predictor:PredictorMultiMaskWrapper,
                 latent_action_encoder:LatentActionEncoderMultiMaskWrapper,
                 state_pooler:AttentivePooler,
                 rewards_decoder:RewardsDec,
                 termination_decoder:TerminDec,
                 action_projector:ActionProjector,
                 
                 # Optimizer
                 optimizer:Optimizer, 
                 scaler:torch.amp.GradScaler, 
                 lr_scheduler:WarmupCosineSchedule, 
                 wd_scheduler:CosineWDSchedule,

                 # Debug
                 tb_logger:TensorboardLogger | None,

                 # Training setting
                 use_amp:bool,
                 amp_dtype:torch.dtype,
                 warmup:int|None=None,
                 clip_grad:float=10.0,

                 ):
        # >> Models setting
        self._context_encoder = context_encoder
        self._target_encoder = target_encoder
        self._predictor = predictor
        self._latent_act_encoder = latent_action_encoder
        self._state_pooler = state_pooler
        self._rewards_decoder = rewards_decoder
        self._termin_decoder = termination_decoder
        self._action_projector = action_projector
        self.modules: List[nn.Module] = [
            self._context_encoder, self._target_encoder, 
            self._predictor, self._latent_act_encoder,
            self._state_pooler, self._rewards_decoder, self._termin_decoder,
            self._action_projector
            ]
        # >> Optimizer setting
        self._optimizer = optimizer
        self._scaler = scaler
        self._lr_scheduler = lr_scheduler
        self._wd_scheduler = wd_scheduler

        # >> Debug setting
        self._tb_logger = tb_logger
        
        # >> Process setting
        self.tubelet_size = self._context_encoder.backbone.tubelet_size
        self.patch_size = self._context_encoder.backbone.patch_size

        # >> training setting
        self._use_amp = use_amp
        self._amp_dtype = amp_dtype
        self._step = 0
        self._warmup = warmup if warmup is not None else self._lr_scheduler.warmup_steps
        self._clip_grad = clip_grad
        self._num_last_frames = -3

        self._action_loss_fn = MSELoss()
        self._reward_loss_fn = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self._termin_loss_fn = nn.BCEWithLogitsLoss()
        
    # Interactive function 
    def step(self):
        pass
    
    def reset(self, sample_obs:torch.Tensor, sample_action:torch.Tensor, buffer_size):
        pass

    def export(self):
        pass

    def encode(self, sample_obs:torch.Tensor):
        pass
    
    def train(self):
        for m in self.modules:
            m.train()
            
    def eval(self):
        for m in self.modules:
            m.eval()
    
    def update(self,
              sample_obs:torch.Tensor, sample_action:torch.Tensor,
              sample_rewards:torch.Tensor, sample_termin:torch.Tensor,
              ):
        '''
        Pseudo code:
        B: batch size
        
        T: frame-level times
        H: frame-level height
        W: frame-level width
        
        t: patch-level times
        h: patch-level height
        w: patch-level width
        P: h*w number of patch
        N: t*P number of all patch on full video
        D: latent dims of patch

        >> Forward process
        context obs = sample_action[:T-self.tubelet_size]
        target encoder = sample_action

        context latent = self._context_encoder(context obs)
        target latent = self._target_encoder(target obs)

        act = self._latent_act_encoder(target latent)
        predicted latent =  self.predictor(z, h, masks_enc, masks_pred, act) #[B P D]

        hat_rewards = self._rewards_decoder(predicted latent)
        hat_termin = self._termin_decoder(predicted latent)
        hat_act = self._action_projector(sample_action)

        >> Loss calculate
        action loss = KL_d loss(hat_act, act)
        rewards loss = loss(hat_rewards, sample_rewards)
        termin loss = loss(hat_termin, sample_termin)
        
        >> Backward
        loss.backward()
        optimizer.step()
        .
        .
        .

        return debug log
        '''
        self.train()
        _new_lr = self._lr_scheduler.step()
        _new_wd = self._wd_scheduler.step()
        B, C, T, H, W = sample_obs.shape
        t = T // self.tubelet_size
        p = (H // self.patch_size) * (W // self.patch_size)
        def build_masks():
            mask = torch.ones((t, p), dtype=torch.int32).to(sample_obs.device)
            mask[-1, :] = 0
            mask = mask.flatten()
            mask_p = torch.argwhere(mask == 0).reshape(1, -1).expand(B, -1)
            mask_e = torch.nonzero(mask).reshape(1, -1).expand(B, -1)
            return [mask_e], [mask_p]
        
        def forward_target(obs):
            z = self._target_encoder(obs)
            return z

        def forward_latent_action(target_z):
            x = [rearrange(_z, "B (t p) D -> B t p D", t=T//self.tubelet_size, p=(H//self.patch_size)*(W//self.patch_size)) for _z in target_z] \
                if isinstance(target_z, list) else \
                rearrange(target_z, "B (t p) D -> B t p D", t=T//self.tubelet_size, p=(H//self.patch_size)*(W//self.patch_size))
            moduls = self._latent_act_encoder.backbone

            for block in moduls.enc_layer:
                x = block(x)  # [B, T, P, D]
            _B, _T, _P, _D = x.shape
            x = x.permute(0, 2, 1, 3).reshape(_B, _T * _P, _D)
            pooled = moduls.attention_pooler(x)
            pooled = pooled.squeeze(1)        # [B, D]
            (z_q, idx), _ = moduls.quant(pooled)
            return z_q, idx
        
        def forward_prediction(obs, mask_e, masks_p, act): 
            z = self._context_encoder(obs, mask_e)
            pred_z = self._predictor(z, None, mask_e, masks_p, act) 
            full_z = torch.concat([z[0], pred_z[0]],dim=1) # [B, ((t-1)*p) + p, D] = [B, (t*p), D]
            return pred_z[0], full_z

        def forward_action_project(real_action):
            return self._action_projector(real_action)
             
        def forward_rewards_decode(predicted_state):
            x = predicted_state[0] if isinstance(predicted_state, list) else predicted_state
            return self._rewards_decoder(self._state_pooler, x)

        def forward_termin_decode(predicted_state):
            x = predicted_state[0] if isinstance(predicted_state, list) else predicted_state
            return self._termin_decoder(self._state_pooler, x)
        
        # loss
        def action_loss_fn(real_act_embed, pseudo_act):
            return self._action_loss_fn(real_act_embed, pseudo_act)
    
        def reward_loss_fn(hat_reward ,reward):
            return self._reward_loss_fn(hat_reward, reward)

        def termin_loss_fn(hat_termin, termin):
            return self._termin_loss_fn(hat_termin, termin)

        self._optimizer.zero_grad()
        # forward
        with torch.amp.autocast(device_type=sample_obs.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
            mask_e, mask_p          = build_masks()
            target_z                = forward_target(sample_obs)
            pseudo_act, quant_idx   = forward_latent_action(target_z)
            _, full_z               = forward_prediction(sample_obs, mask_e, mask_p, pseudo_act)
            hat_rewards             = forward_rewards_decode(full_z[:,self._num_last_frames*p:,])
            hat_termins             = forward_termin_decode(full_z[:,self._num_last_frames*p:,])
            _, proj_quant_vec       = forward_action_project(sample_action[:,-1,:])

            # L2 loss for hat action min distance with quantize vector 
            target_quant_vec = self._latent_act_encoder.backbone.quant.codebook(quant_idx)
            loss_action = action_loss_fn(proj_quant_vec, target_quant_vec)
            loss_reward = reward_loss_fn(hat_rewards, sample_rewards[:,-1])
            loss_termin = termin_loss_fn(hat_termins, sample_termin[:,-1])

        loss = loss_action + loss_reward + loss_termin
        _reward_decoder_norm = 0.
        _termin_decoder_norm = 0.
        _action_projector_norm = 0.
        if self._use_amp:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self._optimizer)
        else:
            loss.backward()

        if (self._step > self._warmup) and (self._clip_grad is not None):
            _reward_decoder_norm = torch.nn.utils.clip_grad_norm_(self._rewards_decoder.parameters(), self._clip_grad)
            _termin_decoder_norm = torch.nn.utils.clip_grad_norm_(self._termin_decoder.parameters(), self._clip_grad)
            _action_projector_norm = torch.nn.utils.clip_grad_norm_(self._action_projector.parameters(), self._clip_grad)
        
        if self._use_amp:
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            self._optimizer.step()

        grad_stats_reward = grad_logger(self._rewards_decoder.named_parameters())
        grad_stats_reward.global_norm = float(_reward_decoder_norm)
        grad_stats_termin = grad_logger(self._termin_decoder.named_parameters())
        grad_stats_termin.global_norm = float(_termin_decoder_norm)
        grad_stats_action = grad_logger(self._action_projector.named_parameters())
        grad_stats_action.global_norm = float(_action_projector_norm)

        self._step +=1
        optim_stats = adamw_logger(self._optimizer)

        if self._tb_logger is not None:
            self._tb_logger.log("WorldModel/action_loss", loss_action.item())
            self._tb_logger.log("WorldModel/reward_loss", loss_reward.item())
            self._tb_logger.log("WorldModel/termination_loss", loss_termin.item())
            self._tb_logger.log("WorldModel/total_loss", loss.item())
           
        return {
            "loss":{
                "total":loss.item(),
                "action":loss_action.item(),
                "reward":loss_reward.item(),
                "termin":loss_termin.item()
            },
            "train_model_state":{
                "reward_decoder":grad_stats_reward,
                "termin_decoder":grad_stats_termin,
                "action_projector":grad_stats_action,
            },
            "optim_state": optim_stats,
            "lr":_new_lr,
            "wd":_new_wd,
        }

