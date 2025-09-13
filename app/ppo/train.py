# train_jepa.py
from __future__ import annotations

import torch
import os, shutil
import numpy as np
from typing import Dict, Any

import yaml
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

from libs.mine_env.utils.actor_critic import DreamerActorCritic
from libs.mine_env.utils.ppo_model import build_ppo 
from libs.mine_env.src.env_factory import build_env
from src.utils.logging import get_logger
from app.ppo.utils import JEPAExtractor

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__)

def _build_policy_kwargs_for_jepa(policy_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build policy_kwargs for SB3 policy so it always uses JEPAExtractor.
    Reads from:
      policy_cfg['features_extractor']['kwargs']  -> passed to JEPAExtractor(...)
      policy_cfg['net_kwargs']                    -> (optional) forwarded to your custom policy if it uses it
      policy_cfg['ortho_init']                   -> (optional) forwarded to policy
    """
    out: Dict[str, Any] = {}

    # --- read FE kwargs from config ---
    fe_cfg = (policy_cfg or {}).get("features_extractor", {})
    fe_kwargs = dict(fe_cfg.get("kwargs") or {})

    # --- required / recommended defaults ---
    fe_kwargs.setdefault("features_dim", 1024)
    fe_kwargs.setdefault("model_name", "vit_large")
    fe_kwargs.setdefault("crop_size", 224)
    fe_kwargs.setdefault("patch_size", 16)
    fe_kwargs.setdefault("num_frames", 16)
    fe_kwargs.setdefault("tubelet_size", 2)
    fe_kwargs.setdefault("uniform_power", False)
    fe_kwargs.setdefault("use_sdpa", True)

    # pooler params with safe defaults
    pp = dict(fe_kwargs.get("pooler_params") or {})
    pp.setdefault("num_heads", 8)
    pp.setdefault("mlp_ratio", 2)
    pp.setdefault("depth", 2)
    pp.setdefault("init_std", 0.02)
    pp.setdefault("qkv_bias", True)
    pp.setdefault("complete_block", True)
    fe_kwargs["pooler_params"] = pp

  
    out["features_extractor_class"] = JEPAExtractor
    out["features_extractor_kwargs"] = fe_kwargs

    # # forward optional policy knobs (if your policy consumes them)
    # if "net_kwargs" in (policy_cfg or {}):
    #     out["net_kwargs"] = dict(policy_cfg["net_kwargs"])
    # if "ortho_init" in (policy_cfg or {}):
    #     out["ortho_init"] = bool(policy_cfg["ortho_init"])

    return out


def main(args: Dict[str, Any], resume_preempt: bool = False):
    # ----- Environment -----
    env_cfg = args["Environment"]
    task = env_cfg["task"]
    num_envs = int(env_cfg["num_envs"])
    seed = int(env_cfg["seed"])
    frame_stack = args["ppo_training"]["policy"]["features_extractor"]["kwargs"]["num_frames"]

    # ----- VecEnv & FrameStack -----
    vec_env = make_vec_env(lambda: build_env(args, seed), n_envs=num_envs)
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)

    # ----- Logging -----
    save_dir = args["logging"]["folder"]
    total_timesteps = int(args["ppo_training"]["training_step"])
    mount_path_env = os.getenv("MOUNT_PATH", "")
    log_dir = os.path.join(mount_path_env, save_dir)

    # ----- Policy & PPO hyper-params -----
    policy_cfg = args["ppo_training"].get("policy", {}) 
    policy_kwargs = _build_policy_kwargs_for_jepa(policy_cfg)

    pn = args["ppo_training"].get("policy_network", {}) 
    algo_gamma = float(pn.get("gamma", 0.99))
    algo_gae_lambda = float(pn.get("gea_lambda", pn.get("gae_lambda", 0.95)))
    algo_ent_coef = float(pn.get("entropy_coef", 0.0))


    model, episode_logger_callback = build_ppo(
        vec_env=vec_env,
        num_envs=num_envs,
        policy=DreamerActorCritic,
        policy_kwargs=policy_kwargs,
        log_dir=log_dir,
        gamma=algo_gamma,
        gae_lambda=algo_gae_lambda,
        ent_coef=algo_ent_coef,
    )

    # Debug prints
    fe = getattr(model.policy, "features_extractor", None)
    print("Features extractor:", fe.__class__.__name__ if fe is not None else None)
    print("MLP extractor:", getattr(model.policy, "mlp_extractor", None))

    # ----- Train -----
    model.learn(total_timesteps=total_timesteps, callback=episode_logger_callback)

    # ----- Save -----
    dummy_config_path = os.path.join(model.logger.dir, "config.yaml")
    model_save_dir = os.path.join(model.logger.dir, "lastest_model")

    if hasattr(args, "config"):
        shutil.copy(args.config, dummy_config_path)
    model.save(model_save_dir)
    vec_env.close()
