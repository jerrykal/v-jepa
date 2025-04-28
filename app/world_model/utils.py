# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys
import os
import yaml
import torch
import pprint
import random
import numpy as np

from tensorboardX import SummaryWriter
from app.world_model.replay_buffer import ReplayBuffer
from src.models.world_models.jepa_world_model import JEPAWorldModel
from src.models.agents.agents import ActorCriticAgent


CONFIG_VERSION = "0.00.0.beta"


def build_world_model(params, action_dims, device)->JEPAWorldModel:
    cfgs_model = params.get("Models").get("WorldModel")
    cfgs_mask = params["mask"]
    
    wm = JEPAWorldModel(
        action_dims=action_dims,
        encoder_name=cfgs_model["encoder_name"],
        image_size=cfgs_model["image_size"],
        patch_size=cfgs_model["patch_size"],
        num_frames=cfgs_model["num_frames"],
        tubelet_size=cfgs_model["tubelet_size"],
        uniform_power=cfgs_model["uniform_power"],

        use_mask_tokens=cfgs_model["use_mask_tokens"],
        pred_embed_dim=cfgs_model["pred_embed_dim"],
        pred_depth=cfgs_model["pred_depth"],
        zero_init_mask_tokens=cfgs_model["zero_init_mask_tokens"],
        loss_exp=cfgs_model["loss_exp"],
        reg_coeff=cfgs_model["reg_coeff"],
        ema=cfgs_model["ema"],

        cfgs_mask=cfgs_mask,
        jepa_pretrain=params.get("pretrain"),
        use_amp=True,
        dtype=torch.float32,
    )
    return wm.to(device=device)

def build_agent(params, action_dim, device)->ActorCriticAgent:
    cfgs_model = params.get("Models").get("Agent")
    cfgs_env = params.get("Environment")
    return ActorCriticAgent(
        feat_dim=sum(cfgs_model["InputFeature"]),
        num_layers=cfgs_model["NumLayers"],
        hidden_dim=cfgs_model["HiddenDim"],
        action_dim=action_dim,
        gamma=float(cfgs_model["Gamma"]),
        lambd=float(cfgs_model["Lambda"]),
        entropy_coef=float(cfgs_model["EntropyCoef"]),
    ).to(device=device)

def build_replay_buffer(params, action_dims, device="cpu"):
    task_parameter = params.get("Environment").get("task_parameter")
    joint_train_agent = params.get("JointTrainAgent")

    return ReplayBuffer(
        obs_shape=(task_parameter.get("image_size")[0], task_parameter.get("image_size")[1], 3),
        action_dim=action_dims,
        num_envs=joint_train_agent.get("NumEnvs"),
        max_length=joint_train_agent.get("BufferMaxLength"),
        warmup_length=joint_train_agent.get("BufferWarmUp"),
        device=device,
    )

def load_config(config_path):
    params = None
    with open(config_path, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print(f"Sysytem config version : {CONFIG_VERSION}")
        print('loaded params...')
        assert "config_version" in params, "config missing config_version"
        assert params["config_version"] == CONFIG_VERSION, "config_version not match"
        print('loaded params success !!')

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    return params

def seed_np_torch(seed=20010105):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
# logger = logging.getLogger()

class Logger():
    def __init__(self) -> None:
        self._init_flag = False

    def init(self, path):
        self.writer = SummaryWriter(logdir=path, flush_secs=1)
        self.tag_step = {}
        self._init_flag = True
    def log(self, tag, value):
        if self._init_flag:
            if tag not in self.tag_step:
                self.tag_step[tag] = 0
            else:
                self.tag_step[tag] += 1
            if "video" in tag:
                self.writer.add_video(tag, value, self.tag_step[tag], fps=15)
            elif "images" in tag:
                self.writer.add_images(tag, value, self.tag_step[tag])
            elif "hist" in tag:
                self.writer.add_histogram(tag, value, self.tag_step[tag])
            else:
                self.writer.add_scalar(tag, value, self.tag_step[tag])
        else:
            raise Exception("Tensorboard Logger is not initiation.")
    def close(self):
        self.writer.close()

## V-JEPA ToDo change to world model
def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')

    epoch = 0
    try:
        epoch = checkpoint['epoch']

        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading predictor
        pretrained_dict = checkpoint['predictor']
        msg = predictor.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained predictor from epoch {epoch} with msg: {msg}')

        # -- loading target_encoder
        if target_encoder is not None:
            print(list(checkpoint.keys()))
            pretrained_dict = checkpoint['target_encoder']
            msg = target_encoder.load_state_dict(pretrained_dict)
            logger.info(
                f'loaded pretrained target encoder from epoch {epoch} with msg: {msg}'
            )

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )