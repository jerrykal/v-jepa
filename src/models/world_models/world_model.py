import torch
import torch.nn as nn

# Modles
from src.models.vision_transformer import VisionTransformer as ViT
from src.models.predictor import VisionTransformerPredictor as PredViT
from src.models.latent_action import LatentActionEncoder as LAE
from src.models.world_models.state_decoder import RewardsDecoder as RewardsDec
from src.models.world_models.state_decoder import TerminationDecoder as TerminDec 
from src.models.world_models.action_projector import ActionProjector as ActPrejector 


class WorldModel():
    '''
    The class is organize the each modules on trianing loop 
    '''
    def __init__(self,
                 # Models
                 context_encoder:ViT,
                 target_encoder:ViT,
                 predictor:PredViT,
                 latent_act_encder:LAE,
                 rewards_decoder:RewardsDec,
                 termination_decoder:TerminDec,
                 action_prejector:ActPrejector,
                 
                 # Optimizer
                 optimizer,
                 scale,

                 # Debug
                 logger,
                 ):
        # >> Models setting
        self._context_encoder = context_encoder
        self._target_encoder = target_encoder
        self._predictor = predictor
        self._latent_act_encder = latent_act_encder
        self._rewards_decoder = rewards_decoder
        self._termination_decoder = termination_decoder
        self._action_prejector = action_prejector
        
        # >> Optimizer setting
        self._optimizer = optimizer
        
        # >> Debug setting
        self._logger = logger
        
        # >> Process setting
        self.tubelet_size = self._context_encoder.tubelet_size
        
    def step(self):
        pass
    
    def reset(self):
        pass

    def encode(self):
        pass

    def train(self,
              sample_obs, sample_action,
              sample_rewards, sample_termin,
              ):
        '''
        Pseudo code:
        B: bastch size
        
        T: frame-level times
        H: frame-level high
        W: frame-level width
        
        t: patch-level times
        h: patch-level high
        w: patch-level width
        P: h*w number of patch
        N: t*P number of all patch on full video
        D: latent dims of patch

        >> Forward process
        context obs = sample_action[:T-self.tubelet_size]
        target encoder = sample_action

        context latent = self._context_encoder(context obs)
        target latent = self._target_encoder(target obs)

        act = self._latent_act_encder(target latent)
        predicted latent =  self.predictor(z, h, masks_enc, masks_pred, act) #[B P D]

        hat_rewards = self._rewards_decoder(predicted latent)
        hat_termin = self._termination_decoder(predicted latent)
        hat_act = self._action_prejector(sample_action)

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
