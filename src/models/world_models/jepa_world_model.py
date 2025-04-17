import copy
import torch

import src.models.vision_transformer as video_vit
import src.models.wm_predictor as vit_predictor
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from typing import List, Tuple
from src.masks.utils import apply_masks
from src.models.utils.losses import SymLogTwoHotLoss
from src.utils.tensors import repeat_interleave_batch
from src.models.attentive_pooler import AttentivePooler
from src.models.vision_transformer import VisionTransformer as ViT
from src.models.world_models.base_world_model import WorldModelBase
from src.models.world_models.DecoderHead import RewardDecoder, TerminationDecoder
from src.masks.multiblock3d import _MaskGenerator,WorldModelMaskGenerator,PrediectFrameMaskGenerator
from src.models.utils.multimask import MultiMaskWrapper, WorldModelPredictorMultiMaskWrapper


def action_onehot_function(actions, action_dims):
    ret = []
    for action, dim in zip(actions, action_dims):
        ret.append(nn.functional.one_hot(action.long(), num_classes=dim).float())
    return ret

def actions2onehot(actions ,action_dims):
    bl_vec = []
    for l_actions in actions: # [B ,L, action]
        l_vec = []
        for action in l_actions: # [L, action]
            l_vec.append(torch.cat(action_onehot_function(action,action_dims), dim=0))
        bl_vec.append(torch.stack(l_vec, dim=0))
    return torch.stack(bl_vec, dim=0)

def init_optimizer(
    models,  # list of torch.nn.Module
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    use_amp=True
):
    param_groups = []
    group_decay = []
    group_no_decay = []

    seen = set()
    for model in models:
        for name, param in model.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))

            if 'bias' not in name and len(param.shape) != 1:
                group_decay.append(param)
            else:
                group_no_decay.append(param)

    param_groups = [
        {'params': group_decay},  # default weight_decay from optimizer
        {'params': group_no_decay, 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if mixed_precision else None
    return optimizer, scaler

class JEPAWorldModel(WorldModelBase):
    def __init__(self,
                 action_dims:int,
                 encoder_name:str="vit_small",
                 image_size:tuple=(224,224),
                 patch_size:int=16,
                 num_frames:int=16,
                 tubelet_size:int=4,
                 uniform_power:bool=False,

                 use_mask_tokens:bool=True,
                 pred_embed_dim:int=384,
                 pred_depth:int=12,
                 zero_init_mask_tokens:bool=True,
                 loss_exp=1.0,
                 reg_coeff=0.0,
                 ema:Tuple[float]=(0.998, 1.0),

                 cfgs_mask:dict={},

                 use_amp=True,
                 dtype=torch.float16,
                 ):
        super().__init__()
        self.use_amp = use_amp
        self.dtype = dtype
        self.reg_coeff = reg_coeff
        self.mixed_precision = (dtype==torch.bfloat16) or (dtype == torch.float16)
        self.action_dims = action_dims
        self.tubelet_size = tubelet_size
        self.img_size = image_size
        self.patch_size = patch_size
        # Context Encoder
        encoder:ViT = video_vit.__dict__[encoder_name](
            img_size=image_size[0],
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            uniform_power=uniform_power
        )
        self.embed_dim = encoder.embed_dim
        self.context_encoder = MultiMaskWrapper(encoder)
        
        # Target Encoder
        self.target_encoder = copy.deepcopy(self.context_encoder)

        # Predictor
        predictor = vit_predictor.__dict__['vit_predictor'](
            img_size=image_size[0],
            use_mask_tokens=use_mask_tokens,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=self.context_encoder.backbone.embed_dim,
            predictor_embed_dim=pred_embed_dim,
            depth=pred_depth,
            num_heads=self.context_encoder.backbone.num_heads,
            uniform_power=uniform_power,
            num_mask_tokens=len(cfgs_mask),
            zero_init_mask_tokens=zero_init_mask_tokens
        )
        self.predictor = WorldModelPredictorMultiMaskWrapper(predictor)

        #frist action token
        self.empty_action = nn.Parameter(torch.zeros(1, 1, pred_embed_dim))
        self.action_encoder = nn.Sequential(
            nn.Linear(sum(action_dims), pred_embed_dim),
            nn.ReLU()
        )

        # Pooler
        self.attentive_pooler = AttentivePooler(
            num_queries=1,
            embed_dim=self.context_encoder.backbone.embed_dim,
            num_heads=self.context_encoder.backbone.num_heads,
            mlp_ratio=4.0,
            depth=1,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            qkv_bias=True,
            complete_block=True,
        )
        # Reward decoder head
        self.reward_decoder = RewardDecoder(
            num_classes=255,
            transformer_hidden_dim=self.context_encoder.backbone.embed_dim
        )

        # Termination decoder head
        self.termination_decoder = TerminationDecoder(
            transformer_hidden_dim=self.context_encoder.backbone.embed_dim
        )

        # Mask
        self.mask_generators:List[_MaskGenerator] = []
        for m in cfgs_mask:
            mask_generator = WorldModelMaskGenerator(
                crop_size=image_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                spatial_pred_mask_scale=m.get('spatial_scale'),
                temporal_pred_mask_scale=m.get('temporal_scale'),
                aspect_ratio=m.get('aspect_ratio'),
                npred=m.get('num_blocks'),
                max_context_frames_ratio=m.get('max_temporal_keep', 1.0),
                max_keep=m.get('max_keep', None),
                use_collect=False
            )
            self.mask_generators.append(mask_generator)
        self.predict_mask_generator = PrediectFrameMaskGenerator(
                crop_size=image_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                spatial_pred_mask_scale=m.get('spatial_scale'),
                temporal_pred_mask_scale=m.get('temporal_scale'),
                aspect_ratio=m.get('aspect_ratio'),
                npred=m.get('num_blocks'),
                max_context_frames_ratio=m.get('max_temporal_keep', 1.0),
                max_keep=m.get('max_keep', None),
                use_collect=False
            )
        #Loss or Optimizer
        # self.mse_loss_func = GeneralizedLpLoss(loss_exp)
        self._loss_exp = loss_exp
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)

        self.optimizer, self.scaler = init_optimizer(    
            [self.context_encoder, self.predictor, self.attentive_pooler, self.reward_decoder, self.termination_decoder],
            mixed_precision=False,
            betas=(0.9, 0.999),
            eps=1e-8,
            zero_init_bias_wd=True,
            use_amp=self.use_amp)
        
        self.ema = ema
    def encode_obs(self,obs):
        return self.context_encoder(obs)
    
    def step(self, embedding, actions):
        # Mask
        batch_size, TP, D = embedding.shape
        T = TP//self.get_num_patches()
        masks_enc, masks_pred = self.predict_mask_generator(batch_size, T)
        _me = masks_enc.cuda()
        _mp = masks_pred.cuda()
        _me = repeat_interleave_batch(_me, batch_size, repeat=1)
        _mp = repeat_interleave_batch(_mp, batch_size, repeat=1)
        
        # Encode action
        actions = actions2onehot(actions, self.action_dims)
        B, A_T, A = actions.shape
        token = self.empty_action.expand(B, -1, -1)
        actions_flat = actions.view(B * A_T, A)
        encoded_action = self.action_encoder(actions_flat)
        encoded_action = encoded_action.view(B, A_T, -1)
        encoded_action = torch.cat([token, encoded_action[:, 1:]], dim=1)
        
        assert A_T == T+1 # Number of action == clip + prediect frame 
        last_z, completed_z=self.predictor.next_frame(embedding, _me, _mp, encoded_action)
        
        feat = self.attentive_pooler(completed_z[0]).squeeze(1)
        reward_hat = self.reward_decoder(feat)
        reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
        
        termination_hat = self.termination_decoder(feat)
        termination_hat = termination_hat > 0
        return completed_z[0], last_z[0], reward_hat, termination_hat

    def get_num_patches(self, num_frames=-1):
        if num_frames == -1:
            num_frames = self.tubelet_size
        return (num_frames//self.tubelet_size) * (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size)

    def get_momentum(self, step, max_steps):
        ema_start, ema_end = self.ema
        progress = min(step / max_steps, 1.0)
        return ema_start + progress * (ema_end - ema_start)
    
    def _forward_target(self, obs, masks_pred):
        """
        Returns list of tensors of shape [B, N, D], one for each
        mask-pred.
        """
        with torch.no_grad():
            feat = self.target_encoder(obs)
            feat = F.layer_norm(feat, (feat.size(-1),))  # normalize over feature-dim  [B, N, D]
            # -- create targets (masked regions of h)
            h = apply_masks(feat, masks_pred, concat=False)
            return h, feat
    
    def _forward_context(self, obs, target_feat, masks_enc, masks_pred, encoded_action):
        """
        Returns list of tensors of shape [B, N, D], one for each
        mask-pred.
        """
        z = self.context_encoder(obs, masks_enc)
        z, compelet_z = self.predictor(z, target_feat, masks_enc, masks_pred, encoded_action)

        return z, compelet_z
    
    def _recon_loss_func(self, z, h, masks_pred):
        loss = 0.
        # Compute loss and accumulate for each mask-enc/mask-pred pair
        for zi, hi in zip(z, h):
            loss += torch.mean(torch.abs(zi - hi)**self._loss_exp) / self._loss_exp
        loss /= len(masks_pred)
        return loss
    
    def _reg_fn(self, z):
        return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)
    
    def update(self, obs, actions, reward, termination, logger=None, log_video=False, **kwargs):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        visual_obs = obs.cpu().float().detach()
        self.target_encoder.eval()
        try:
            step = kwargs["step"]
            max_steps = kwargs["max_steps"]
        except KeyError as e:
            raise KeyError(f"Missing required argument: {e}")

        with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.use_amp):
            # Encode action
            actions = actions2onehot(actions[:,::self.tubelet_size], self.action_dims)
            B, T, A = actions.shape 
            token = self.empty_action.expand(B, -1, -1)
            actions_flat = actions.view(B * T, A)
            encoded_action = self.action_encoder(actions_flat)
            encoded_action = encoded_action.view(B, T, -1)
            encoded_action = torch.cat([token, encoded_action[:, :-1]], dim=1) 

            # update mask
            collated_masks_enc, collated_masks_pred  = [], []
            for i, mask_generator in enumerate(self.mask_generators):
                masks_enc, masks_pred = mask_generator(batch_size)
                collated_masks_enc.append(masks_enc)
                collated_masks_pred.append(masks_pred)

            # Put each mask-enc/mask-pred pair on the GPU and reuse the
            # same mask pair for each clip
            _masks_enc, _masks_pred = [], []
            for _me, _mp in zip(collated_masks_enc, collated_masks_pred):
                _me = _me.cuda()
                _mp = _mp.cuda()
                _me = repeat_interleave_batch(_me, batch_size, repeat=1)
                _mp = repeat_interleave_batch(_mp, batch_size, repeat=1)
                _masks_enc.append(_me)
                _masks_pred.append(_mp)

            # JEPA Reconstruction Loss
            loss_jepa, loss_reg = 0., 0.
            h, _ = self._forward_target(obs, _masks_pred)
            z, completed_z = self._forward_context(obs, h, _masks_enc, _masks_pred, encoded_action)
            loss_jepa = self._recon_loss_func(z, h, _masks_pred)  # jepa prediction loss
            pstd_z = self._reg_fn(z)  # predictor variance across patches
            loss_reg += torch.mean(F.relu(1.-pstd_z))
            
            temporl_z = completed_z[-1] #Temporl mask predict
            feat = self.attentive_pooler(temporl_z).squeeze(1)
            reward_hat = self.reward_decoder(feat)
            termination_hat = self.termination_decoder(feat)

            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward[:,-1].squeeze())
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination[:,-1].squeeze())

            loss = loss_jepa + self.reg_coeff * loss_reg #+ reward_loss + termination_loss
            # Step 2. Backward & step
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            
            # _enc_norm, _pred_norm = 0., 0.
            # if (epoch > warmup) and (clip_grad is not None):
            #     _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
            #     _pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)

            if self.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            # grad_stats = grad_logger(self.context_encoder.named_parameters())
            # grad_stats.global_norm = float(_enc_norm)
            # grad_stats_pred = grad_logger(predictor.named_parameters())
            # grad_stats_pred.global_norm = float(_pred_norm)
            # optim_stats = self.adamw_logger(self.optimizer)

            # Step 3. momentum update of target encoder
            m = self.get_momentum(step=step,max_steps=max_steps)
            with torch.no_grad():
                for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

            if logger is not None:
                logger.log("WorldModel/loss_jepa", loss_jepa.item())
                logger.log("WorldModel/loss_reg", loss_reg.item())
                logger.log("WorldModel/reward_loss", reward_loss.item())
                logger.log("WorldModel/termination_loss", termination_loss.item())
                logger.log("WorldModel/total_loss", loss.item())
                if log_video:
                    visual_obs = rearrange(visual_obs, "B C T H W ->B T C H W")
                    logger.log("Recon/world_sample_video", torch.clamp(visual_obs[::batch_size//2], 0, 1).numpy())
            
def test_jepa_world_model():
    import numpy as np
    batch_size = 2
    time_steps = 16
    img_size = (224, 224)
    patch_size = 16
    tubelet_size = 2
    channels = 3

    cfgs_mask = [{
        "spatial_scale": (0.15, 0.15),
        "temporal_scale": (1.0, 1.0),
        "aspect_ratio": (0.75, 1.5),
        "num_blocks": 8,
        "max_temporal_keep": 1.0,
        "max_keep": None,
    },
    {
        "spatial_scale": (0.7, 0.7),
        "temporal_scale": (1.0, 1.0),
        "aspect_ratio": (0.75, 1.5),
        "num_blocks": 2,
        "max_temporal_keep": 1.0,
        "max_keep": None,
        
    }]
    action_dims=[12,3]
    # >>> Build up replay buffer
    from app.world_model.replay_buffer import ReplayBuffer
    replay_buffer = ReplayBuffer(
        obs_shape=(img_size[0], img_size[1], 3),
        action_dim=action_dims,
        num_envs=1,
        max_length=100000,
        warmup_length=500,
        device='cpu',
    )
    replay_buffer.load_buffer("test.pkl")
    
    model = JEPAWorldModel(
        action_dims=action_dims,
        encoder_name="vit_small",
        image_size=img_size,
        patch_size=patch_size,
        num_frames=time_steps,
        tubelet_size=tubelet_size,
        cfgs_mask=cfgs_mask,
        use_amp=False,
        dtype=torch.float32,
    ).cuda()

    step = 1000
    max_steps = 100000
    
    obs, actions, rewards, terminations = replay_buffer.sample(batch_size, 0, 16)
    model.update(
        obs=obs.cuda(),
        actions=actions.cuda(),
        reward=rewards.cuda(),
        termination=terminations.cuda(),
        step=step,
        max_steps=max_steps
    )

