from src.models.world_models.world_model import WorldModel
from src.models.agents.agent import ActorCriticAgent
from pathlib import Path
from typing import Optional, Dict, Any
import time, random, tempfile
import numpy as np
import torch
import torch.distributed as dist

def _unwrap_ddp(m):
    return getattr(m, "module", m)

def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def _get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    for group in optimizer.param_groups:
        if "lr" in group:
            return float(group["lr"])
    return float("nan")

class CheckpointIO:
    @staticmethod
    def build_state_dict(
        world: WorldModel,
        agent: ActorCriticAgent,
    ) -> Dict[str, Any]:
        # --- helpers ---
        def _ema_state(ema_obj):
            if hasattr(ema_obj, "state_dict"):
                try:
                    return ema_obj.state_dict()
                except Exception:
                    pass
            state = {}
            for k in ("value", "avg", "decay", "initialized"):
                if hasattr(ema_obj, k):
                    v = getattr(ema_obj, k)
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().item() if v.ndim == 0 else v.detach().cpu().tolist()
                    elif isinstance(v, np.ndarray):
                        v = v.tolist()
                    state[k] = v
            return state or None

        # --- unwrap modules (以防外面包 DDP) ---
        ctx_enc = _unwrap_ddp(world._context_encoder)
        tgt_enc = _unwrap_ddp(world._target_encoder)
        predictor = _unwrap_ddp(world._predictor)
        lae = _unwrap_ddp(world._latent_act_encder)
        pooler = _unwrap_ddp(world._state_pooler)
        rew_dec = _unwrap_ddp(world._rewards_decoder)
        ter_dec = _unwrap_ddp(world._termin_decoder)
        act_proj = _unwrap_ddp(world._action_projector)

        ac_actor = _unwrap_ddp(agent.actor)
        ac_critic = _unwrap_ddp(agent.critic)
        ac_slow_critic = _unwrap_ddp(agent.slow_critic)

        ckpt: Dict[str, Any] = {
            "version": 1,
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),

            # ---- World Model ----
            "world_model": {
                "model": {
                    "context_encoder": ctx_enc.state_dict(),
                    "target_encoder": tgt_enc.state_dict(),
                    "predictor": predictor.state_dict(),
                    "latent_action_encoder": lae.state_dict(),
                    "state_pooler": pooler.state_dict(),
                    "rewards_decoder": rew_dec.state_dict(),
                    "termination_decoder": ter_dec.state_dict(),
                    "action_projector": act_proj.state_dict(),
                },
                "optimizer": world._optimizer.state_dict() if getattr(world, "_optimizer", None) is not None else None,
                "scaler": world._scaler.state_dict() if getattr(world, "_scaler", None) is not None else None,
                "schedulers": {
                    "lr": world._lr_scheduler.state_dict() if getattr(world, "_lr_scheduler", None) is not None else None,
                    "wd": world._wd_scheduler.state_dict() if getattr(world, "_wd_scheduler", None) is not None else None,
                },
                "train_cfg": {
                    "use_amp": bool(world._use_amp),
                    "amp_dtype": str(world._amp_dtype).replace("torch.", ""),
                    "clip_grad": float(world._clip_grad) if world._clip_grad is not None else None,
                    "warmup": int(world._warmup) if world._warmup is not None else None,
                    "num_last_frames": int(world._num_last_frames),
                    "tubelet_size": int(world.tubelet_size),
                    "patch_size": int(world.patch_size),
                    "global_step": int(world._step),
                },
                "meta": {
                    "lr": _get_current_lr(world._optimizer) if getattr(world, "_optimizer", None) is not None else None,
                },
            },

            # ---- Actor-Critic Agent ----
            "agent": {
                "model": {
                    "actor": ac_actor.state_dict(),
                    "critic": ac_critic.state_dict(),
                    "slow_critic": ac_slow_critic.state_dict(),
                },
                "optimizer": agent.optimizer.state_dict() if getattr(agent, "optimizer", None) is not None else None,
                "scaler": agent.scaler.state_dict() if getattr(agent, "scaler", None) is not None else None,
                "ema_state": {
                    "lowerbound": _ema_state(agent.lowerbound_ema),
                    "upperbound": _ema_state(agent.upperbound_ema),
                },
            },

            # # ---- RNG ----
            # "rng_state": {
            #     "python": random.getstate(),
            #     "numpy": np.random.get_state(),
            #     "torch": torch.get_rng_state().tolist(),
            #     "cuda": [s.tolist() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
            # },
        }

        return ckpt

    @staticmethod
    def atomic_save(obj: Dict[str, Any], path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
            tmp_path = Path(tmp.name)
            torch.save(obj, tmp_path)
        tmp_path.replace(path)

    @staticmethod
    def save(
        world: WorldModel,
        agent: ActorCriticAgent,
        path: str | Path,
        only_main_process: bool = True,
        logger=None,
    ):
        if only_main_process and not is_main_process():
            return
        try:
            ckpt = CheckpointIO.build_state_dict(
                world=world,
                agent=agent
            )
            CheckpointIO.atomic_save(ckpt, path)
            if logger: logger.info(f"Checkpoint saved to: {path}")
        except Exception as e:
            if logger: logger.info(f"Encountered exception when saving checkpoint: {e}")
            else: print(f"[Checkpoint Save Error] {e}")

    @staticmethod
    def load(
        world: WorldModel,
        agent: ActorCriticAgent,
        path: str | Path,
    ) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location="cpu")

        # ---- unwrap ----
        ctx_enc = _unwrap_ddp(world._context_encoder)
        tgt_enc = _unwrap_ddp(world._target_encoder)
        predictor = _unwrap_ddp(world._predictor)
        lae = _unwrap_ddp(world._latent_act_encder)
        pooler = _unwrap_ddp(world._state_pooler)
        rew_dec = _unwrap_ddp(world._rewards_decoder)
        ter_dec = _unwrap_ddp(world._termin_decoder)
        act_proj = _unwrap_ddp(world._action_projector)

        ac_actor = _unwrap_ddp(agent.actor)
        ac_critic = _unwrap_ddp(agent.critic)
        ac_slow_critic = _unwrap_ddp(agent.slow_critic)

        # ---- load world model ----
        wm = ckpt.get("world_model", {})
        wm_model = wm.get("model", {})
        if wm_model:
            if wm_model.get("context_encoder") is not None:
                ctx_enc.load_state_dict(wm_model["context_encoder"], strict=True)
            if wm_model.get("target_encoder") is not None:
                tgt_enc.load_state_dict(wm_model["target_encoder"], strict=True)
            if wm_model.get("predictor") is not None:
                predictor.load_state_dict(wm_model["predictor"], strict=True)
            if wm_model.get("latent_action_encoder") is not None:
                lae.load_state_dict(wm_model["latent_action_encoder"], strict=True)
            if wm_model.get("state_pooler") is not None:
                pooler.load_state_dict(wm_model["state_pooler"], strict=True)
            if wm_model.get("rewards_decoder") is not None:
                rew_dec.load_state_dict(wm_model["rewards_decoder"], strict=True)
            if wm_model.get("termination_decoder") is not None:
                ter_dec.load_state_dict(wm_model["termination_decoder"], strict=True)
            if wm_model.get("action_projector") is not None:
                act_proj.load_state_dict(wm_model["action_projector"], strict=True)

        # optimizer / scaler / schedulers
        if wm.get("optimizer") is not None and getattr(world, "_optimizer", None) is not None:
            world._optimizer.load_state_dict(wm["optimizer"])
        if wm.get("scaler") is not None and getattr(world, "_scaler", None) is not None:
            world._scaler.load_state_dict(wm["scaler"])
        sched = wm.get("schedulers", {})
        if sched:
            if sched.get("lr") is not None and getattr(world, "_lr_scheduler", None) is not None:
                world._lr_scheduler.load_state_dict(sched["lr"])
            if sched.get("wd") is not None and getattr(world, "_wd_scheduler", None) is not None:
                world._wd_scheduler.load_state_dict(sched["wd"])

        # train cfg / step
        train_cfg = wm.get("train_cfg", {})
        if "global_step" in train_cfg:
            world._step = int(train_cfg["global_step"])

        # ---- load agent ----
        ag = ckpt.get("agent", {})
        ag_model = ag.get("model", {})
        if ag_model:
            if ag_model.get("actor") is not None:
                ac_actor.load_state_dict(ag_model["actor"], strict=True)
            if ag_model.get("critic") is not None:
                ac_critic.load_state_dict(ag_model["critic"], strict=True)
            if ag_model.get("slow_critic") is not None:
                ac_slow_critic.load_state_dict(ag_model["slow_critic"], strict=True)

        if ag.get("optimizer") is not None and getattr(agent, "optimizer", None) is not None:
            agent.optimizer.load_state_dict(ag["optimizer"])
        if ag.get("scaler") is not None and getattr(agent, "scaler", None) is not None:
            agent.scaler.load_state_dict(ag["scaler"])

        # EMA
        def _ema_load(ema_obj, state):
            if not state:
                return
            if hasattr(ema_obj, "load_state_dict"):
                try:
                    ema_obj.load_state_dict(state)
                    return
                except Exception:
                    pass
            for k, v in state.items():
                if hasattr(ema_obj, k):
                    setattr(ema_obj, k, v)

        ema_state = ag.get("ema_state", {})
        _ema_load(agent.lowerbound_ema, ema_state.get("lowerbound"))
        _ema_load(agent.upperbound_ema, ema_state.get("upperbound"))

        return ckpt