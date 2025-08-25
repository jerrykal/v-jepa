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

def build_masks(B:int, t:int, p:int, device):
    mask = torch.ones((t, p), dtype=torch.int32).to(device)
    mask[-1, :] = 0
    mask = mask.flatten()
    mask_p = torch.argwhere(mask == 0).reshape(1, -1).expand(B, -1)
    mask_e = torch.nonzero(mask).reshape(1, -1).expand(B, -1)
    return [mask_e], [mask_p]

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
                 action_dims:list,
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
        self.train_modules: List[nn.Module] = [
            self._state_pooler, self._rewards_decoder, self._termin_decoder,
            self._action_projector
            ]
        self.modules: List[nn.Module] = [
            self._context_encoder, self._target_encoder, 
            self._predictor, self._latent_act_encoder,
            self._state_pooler, self._rewards_decoder, self._termin_decoder,
            self._action_projector
            ]
        self.action_dims = action_dims

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
        
        # >> Training setting
        self._use_amp = use_amp
        self._amp_dtype = amp_dtype
        self._step = 0
        self._warmup = warmup if warmup is not None else self._lr_scheduler.warmup_steps
        self._clip_grad = clip_grad
        self._num_last_frames = -3

        self._action_loss_fn = MSELoss()
        self._reward_loss_fn = SymLogTwoHotLoss(num_classes=rewards_decoder.num_classes, lower_bound=-20, upper_bound=20).cuda()
        self._termin_loss_fn = nn.BCEWithLogitsLoss()

        # >> Variables
        # Private:
        self._current_latent:torch.Tensor = None
        self._i = 0
        self._B = 0
        self._t = 0
        self._p = 0

        # Public: 
        self.video_feature_dim = self._context_encoder.backbone.embed_dim 
        self.imagination_batch_size = -1
        self.imagination_batch_length = -1
        self.latent_buffer = None
        self.action_buffer = None
        self.reward_hat_buffer = None
        self.termination_hat_buffer = None

    # Interactive function 
    def step(self, action):
        with torch.no_grad():
            (act, _), _         = self._action_projector(action)
            z                   = self._current_latent[:,self._p:] # [B (1:T)*P D]
            masks_e, masks_p    = build_masks(B=self._B, t=self._t, p=self._p, device=self._current_latent.device)
            pred_z              = self._predictor(z, None, masks_e, masks_p, act)[0] # list{[B P D]}[0]
        
        self.latent_buffer[:,self._i:self._i+2] = self._current_latent[:,-2*self._p:].contiguous().view(self._B, 2, self._p, self.video_feature_dim) # [B i:i+1 P D]
        self.action_buffer[:,self._i] = action

        self._current_latent = torch.concat([self._current_latent[:,self._p:], pred_z],dim=1)
        self.target_latent = torch.concat([self.target_latent[:,self._p:], pred_z],dim=1)
        pooled_z = self._state_pooler(self._current_latent[:,self._num_last_frames*self._p:,]).squeeze(1) 
        
        self.reward_hat_buffer[:,self._i] = self._reward_loss_fn.decode(self._rewards_decoder(pooled_z)).squeeze(-1) 
        self.termination_hat_buffer[:,self._i] = (self._termin_decoder(pooled_z) > 0)

        self._bump_index()
        return StateFeature(x=self.target_latent, t=self._t, p=self._p)

    def reset(self, sample_obs:torch.Tensor, imagination_batch_size:int, imagination_batch_length:int):
        with torch.no_grad():
            B, C, T, H, W = sample_obs.shape
            self._B = B
            self._t = T // self.tubelet_size
            self._p = (H // self.patch_size) * (W // self.patch_size)
            self._i = 0
            self._init_buffer(imagination_batch_size, imagination_batch_length)
            self._current_latent = self._context_encoder(sample_obs)
            self.target_latent = self._target_encoder(sample_obs)
        return StateFeature(x=self.target_latent, t=self._t, p=self._p)

    def export(self, device):
        export_state = StateFeature.from_time_patches(self.latent_buffer.to(device))
        export_action = self.action_buffer.to(device)
        export_rewards = self.reward_hat_buffer.to(device)
        export_terminations = self.termination_hat_buffer.to(device)

        return export_state, export_action, export_rewards, export_terminations
    
    def _bump_index(self):
        if self.imagination_batch_length <= 0:
            raise RuntimeError("Buffer length is zero; call reset/_init_buffer first.")
        self._i = (self._i + 1) % self.imagination_batch_length

    def _init_buffer(self, imagination_batch_size, imagination_batch_length):
            '''
                This can slightly improve the efficiency of imagination data But may vary across different machines
            '''
            if self.imagination_batch_size != imagination_batch_size or self.imagination_batch_length != imagination_batch_length:
                logger.info(f"init_imagination_buffer: {imagination_batch_size}x{imagination_batch_length}@{self._amp_dtype}")

                self.imagination_batch_size = imagination_batch_size
                self.imagination_batch_length = imagination_batch_length

                latent_size = (imagination_batch_size, imagination_batch_length+1 , self._p, self.video_feature_dim) 
                action_size = (imagination_batch_size, imagination_batch_length   , len(self.action_dims))
                scalar_size = (imagination_batch_size, imagination_batch_length  )

                self.latent_buffer = torch.zeros(latent_size, dtype=self._amp_dtype, device="cuda") 
                self.action_buffer = torch.zeros(action_size, dtype=self._amp_dtype, device="cuda")
                self.reward_hat_buffer = torch.zeros(scalar_size, dtype=self._amp_dtype, device="cuda")
                self.termination_hat_buffer = torch.zeros(scalar_size, dtype=self._amp_dtype, device="cuda")

    def encode(self, sample_obs:torch.Tensor):
        B, C, T, H, W = sample_obs.shape
        t = T // self.tubelet_size
        p = (H // self.patch_size) * (W // self.patch_size)
        return StateFeature(x=self._target_encoder(sample_obs), t=t, p=p)
    
    def train(self):
        for m in self.train_modules:
            m.train()
            
    def eval(self):
        for m in self.modules:
            m.eval()
    
    def update(self,
              sample_obs:torch.Tensor, sample_action:torch.Tensor,
              sample_rewards:torch.Tensor, sample_termin:torch.Tensor,
              ):
        """
        One training step for the World Model.
        Args:
            sample_obs:     [B, C, T, H, W] - input video frames
            sample_action:  [B, T, ...]     - action sequence
            sample_rewards: [B, T]          - reward targets
            sample_termin:  [B, T]          - termination targets
        Returns:
            dict of loss values, grad stats, optimizer state, lr/wd logs
        """
        # >> switch to train mode 
        self.train()

         # step LR and WD schedulers
        _new_lr = self._lr_scheduler.step()
        _new_wd = self._wd_scheduler.step()

        # get shapes
        B, C, T, H, W = sample_obs.shape
        t = T // self.tubelet_size
        p = (H // self.patch_size) * (W // self.patch_size)

        # >> helper: run target encoder
        def forward_target(obs):
            z = self._target_encoder(obs)
            return z
        
        # >> helper: run latent action encoder & quantizer
        def forward_latent_action(target_z):
            x = [rearrange(_z, "B (t p) D -> B t p D", t=T//self.tubelet_size, p=(H//self.patch_size)*(W//self.patch_size)) for _z in target_z] \
                if isinstance(target_z, list) else \
                rearrange(target_z, "B (t p) D -> B t p D", t=T//self.tubelet_size, p=(H//self.patch_size)*(W//self.patch_size))
            moduls = self._latent_act_encoder.backbone 

            # pass through temporal encoder blocks
            for block in moduls.enc_layer:
                x = block(x)  # [B, T, P, D]
            _B, _T, _P, _D = x.shape
            
            # flatten [B, T, P, D] -> [B, T*P, D]
            x = x.permute(0, 2, 1, 3).reshape(_B, _T * _P, _D)
            pooled = moduls.attention_pooler(x)
            pooled = pooled.squeeze(1)        # [B, D]
            (z_q, idx), _ = moduls.quant(pooled)
            return z_q, idx
        
        # >> helper: context encoder + predictor 
        def forward_prediction(obs, mask_e, masks_p, act): 
            z = self._context_encoder(obs, mask_e)
            pred_z = self._predictor(z, None, mask_e, masks_p, act) 
            full_z = torch.concat([z[0], pred_z[0]],dim=1) # [B, ((t-1)*p) + p, D] = [B, (t*p), D]
            return pred_z[0], full_z
        
        # >> helper: project real action to latent space
        def forward_action_project(real_action):
            return self._action_projector(real_action)
        
        def forward_pooling(predicted_state):
            x = predicted_state[0] if isinstance(predicted_state, list) else predicted_state
            return self._state_pooler(x).squeeze(1) # [B N D] -> [B Q D]

        # >> helper: decode reward from latent state
        def forward_rewards_decode(x):
            return self._rewards_decoder(x)
        
        # >> helper: decode termination from latent state
        def forward_termin_decode(x):
            return self._termin_decoder(x)
        
        # >> helper: loss wrappers 
        def action_loss_fn(real_act_embed, pseudo_act):
            return self._action_loss_fn(real_act_embed, pseudo_act)
    
        def reward_loss_fn(hat_reward ,reward):
            return self._reward_loss_fn(hat_reward, reward)

        def termin_loss_fn(hat_termin, termin):
            return self._termin_loss_fn(hat_termin, termin)

        # >> forward pass
        self._optimizer.zero_grad()
        with torch.amp.autocast(device_type=sample_obs.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
            mask_e, mask_p          = build_masks(B=B, t=t, p=p, device=sample_obs.device)
            target_z                = forward_target(sample_obs)
            pseudo_act, quant_idx   = forward_latent_action(target_z)
            _, full_z               = forward_prediction(sample_obs, mask_e, mask_p, pseudo_act)
            pooled_z                = forward_pooling(full_z[:,self._num_last_frames*p:,])
            hat_rewards             = forward_rewards_decode(pooled_z)
            hat_termins             = forward_termin_decode(pooled_z)
            _, proj_quant_vec       = forward_action_project(sample_action[:,-1,:])

            # compute action target from quantizer codebook
            target_quant_vec = self._latent_act_encoder.backbone.quant.codebook(quant_idx) 
            loss_action = action_loss_fn(proj_quant_vec, target_quant_vec)
            loss_reward = reward_loss_fn(hat_rewards, sample_rewards[:,-1])
            loss_termin = termin_loss_fn(hat_termins, sample_termin[:,-1])

        # total loss
        loss = loss_action + loss_reward + loss_termin

         # >> backward pass 
        _reward_decoder_norm = 0.
        _termin_decoder_norm = 0.
        _action_projector_norm = 0.
        if self._use_amp:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self._optimizer)
        else:
            loss.backward()
        
        # gradient clipping (only after warmup)
        if (self._step > self._warmup) and (self._clip_grad is not None):
            _reward_decoder_norm = torch.nn.utils.clip_grad_norm_(self._rewards_decoder.parameters(), self._clip_grad)
            _termin_decoder_norm = torch.nn.utils.clip_grad_norm_(self._termin_decoder.parameters(), self._clip_grad)
            _action_projector_norm = torch.nn.utils.clip_grad_norm_(self._action_projector.parameters(), self._clip_grad)
        
        # optimizer step
        if self._use_amp:
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            self._optimizer.step()

        # >> logging grad stats 
        grad_stats_reward = grad_logger(self._rewards_decoder.named_parameters())
        grad_stats_reward.global_norm = float(_reward_decoder_norm)
        grad_stats_termin = grad_logger(self._termin_decoder.named_parameters())
        grad_stats_termin.global_norm = float(_termin_decoder_norm)
        grad_stats_action = grad_logger(self._action_projector.named_parameters())
        grad_stats_action.global_norm = float(_action_projector_norm)

        # >> step counter + optimizer logging
        self._step +=1
        optim_stats = adamw_logger(self._optimizer)

        if self._tb_logger is not None:
            self._tb_logger.log("WorldModel/action_loss", loss_action.item())
            self._tb_logger.log("WorldModel/reward_loss", loss_reward.item())
            self._tb_logger.log("WorldModel/termination_loss", loss_termin.item())
            self._tb_logger.log("WorldModel/total_loss", loss.item())

        # >> return summary 
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
            "optim_stats": optim_stats,
            "lr":_new_lr,
            "wd":_new_wd,
        }

