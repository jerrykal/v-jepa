# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# import warnings
# warnings.filterwarnings('ignore')

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import time
import numpy as np

import torch
from collections import deque
import torch.nn.functional as F
import torch.multiprocessing as mp

# RL Training Environment
from libs import env_factory

from einops import rearrange
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    get_logger,
    grad_logger,
    adamw_logger,
    AverageMeter,
    TensorboardLogger)
from torch.nn.parallel import DistributedDataParallel
from src.utils.tensors import repeat_interleave_batch
from src.utils.action_parser import ActionParser
from app.world_model.ckpt_io import CheckpointIO

from app.world_model.utils import (
    init_replay_buffer,
    init_world_model,
    init_agent
)

# --
log_timings = True
log_freq = 500
checkpoint_freq = 500
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__)

IMAGE_CHANNEL = 3

def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    cfgs_meta = args.get('meta')
    load_model = cfgs_meta.get('load_checkpoint') or resume_preempt
    ckpt_file = cfgs_meta.get('read_checkpoint', None)
    seed = cfgs_meta.get('seed', _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get('save_every_freq', -1)
    # skip_batches = cfgs_meta.get('skip_batches', -1)
    # use_sdpa = cfgs_meta.get('use_sdpa', False)
    which_dtype = cfgs_meta.get('dtype')
    logger.info(f'{which_dtype=}')
    if which_dtype.lower() == 'bfloat16':
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == 'float16':
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False
    pre_train_model = cfgs_meta.get('pre_train_model', None)

    # # -- MASK
    # cfgs_mask = args.get('mask')

    # -- WORLD MODEL
    cfgs_wm = args.get('world_model')
    video_model_params = cfgs_wm["video_model_params"]
    latent_action_enc_params = cfgs_wm["latent_action_enc_params"]
    state_decoder_params = cfgs_wm["state_decoder_params"]
    action_projector_params = cfgs_wm["action_projector_params"]
    optimizer_params = cfgs_wm["optimizer_params"]
    pretrained_model_path = cfgs_wm["pretrained_model_path"]
    fine_tune = cfgs_wm["fine_tune"]

    # -- AGENT
    cfgs_agent = args.get('agent')
    feat_len = cfgs_agent["feat_len"]
    feat_dim = cfgs_agent["feat_dim"]
    num_layers = cfgs_agent["num_layers"]
    hidden_dim = cfgs_agent["hidden_dim"]
    action_dim = cfgs_agent["action_dim"]
    gamma = cfgs_agent["gamma"]
    lambd = cfgs_agent["lambd"]
    entropy_coef = cfgs_agent["entropy_coef"]

    # -- ENV
    cfgs_env = args.get('env')
    env_params = cfgs_env["env_params"]
    frame_skip = cfgs_env["frame_skip"]
    maxpooling = cfgs_env["maxpooling"]

    # -- REPALY BUFFER
    cfgs_rb = args.get('replay_buffer')
    max_length = cfgs_rb["max_length"]
    warmup_length = cfgs_rb["warmup_length"]
    store_on_gpu = cfgs_rb["store_on_gpu"]
    export_data_path = cfgs_rb["export_data_path"]
    pre_collected_data_path = cfgs_rb["pre_collected_data_path"]

    # -- OPTIMIZATION
    cfgs_opt = args.get('optimization')
    max_steps = cfgs_opt["max_steps"]
    num_envs = cfgs_opt["num_envs"]

    train_setting = cfgs_opt["train"]
    batch_size = train_setting["batch_size"]
    seq_length = train_setting["seq_length"]
    save_interval = train_setting["save_interval"]

    demonstration_setting = cfgs_opt["demonstration"]
    demon_enable = demonstration_setting["enabled"]
    demon_batch_size = demonstration_setting["batch_size"]

    imagination_setting = cfgs_opt["imagination"]
    imagination_batch_size = imagination_setting["batch_size"]
    imagination_seq_length = imagination_setting["seq_length"]
    imagination_context_length = imagination_setting["context_length"]
    imagination_demon_batch_size = imagination_setting["demo_batch_size"]

    update_setting = cfgs_opt["update"]
    world_model_interval = update_setting["world_model_interval"]
    agent_interval = update_setting["agent_interval"]

    # # -- DATA
    # cfgs_data = args.get('data')
    # dataset_type = cfgs_data.get('dataset_type', 'videodataset')
    # mask_type = cfgs_data.get('mask_type', 'multiblock3d')
    # dataset_paths = cfgs_data.get('datasets', [])
    # datasets_weights = cfgs_data.get('datasets_weights', None)
    # if datasets_weights is not None:
    #     assert len(datasets_weights) == len(dataset_paths), 'Must have one sampling weight specified for each dataset'
    # batch_size = cfgs_data.get('batch_size')
    # num_clips = cfgs_data.get('num_clips')
    # num_frames = cfgs_data.get('num_frames')
    # tubelet_size = cfgs_data.get('tubelet_size')
    # sampling_rate = cfgs_data.get('sampling_rate')
    # duration = cfgs_data.get('clip_duration', None)
    # crop_size = cfgs_data.get('crop_size', 224)
    # patch_size = cfgs_data.get('patch_size')
    # pin_mem = cfgs_data.get('pin_mem', False)
    # num_workers = cfgs_data.get('num_workers', 1)
    # filter_short_videos = cfgs_data.get('filter_short_videos', False)
    # decode_one_clip = cfgs_data.get('decode_one_clip', True)
    # log_resource_util_data = cfgs_data.get('log_resource_utilization', False)

    # # -- LOSS
    # cfgs_loss = args.get('loss')
    # loss_exp = cfgs_loss.get('loss_exp')
    # reg_coeff = cfgs_loss.get('reg_coeff')
    # quant_coeff = cfgs_loss.get('quant_coeff')

    # # -- OPTIMIZATION
    # cfgs_opt = args.get('optimization')
    # ipe = cfgs_opt.get('ipe', None)
    # ipe_scale = cfgs_opt.get('ipe_scale', 1.0)
    # clip_grad = cfgs_opt.get('clip_grad', None)
    # wd = float(cfgs_opt.get('weight_decay'))
    # final_wd = float(cfgs_opt.get('final_weight_decay'))
    # num_epochs = cfgs_opt.get('epochs')
    # warmup = cfgs_opt.get('warmup')
    # start_lr = cfgs_opt.get('start_lr')
    # lr = cfgs_opt.get('lr')
    # final_lr = cfgs_opt.get('final_lr')
    # ema = cfgs_opt.get('ema')
    # betas = cfgs_opt.get('betas', (0.9, 0.999))
    # eps = cfgs_opt.get('eps', 1.e-8)

    # -- LOGGING
    cfgs_logging = args.get('logging')
    folder = cfgs_logging.get('folder')
    tag = cfgs_logging.get('write_tag')

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_file = f'{tag}-latest.pth.tar'
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        load_path = os.path.join(folder, ckpt_file) if ckpt_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- init logger
    logger.info('Initializing loader...')
    tb_logger = TensorboardLogger(path=folder)
    
    ## -- meter of world model loss 
    wm_total_loss_meter = AverageMeter()
    wm_action_loss_meter = AverageMeter()
    wm_reward_loss_meter = AverageMeter()
    wm_termin_loss_meter = AverageMeter()

    ## -- meter of agent loss 
    agent_total_loss_meter = AverageMeter()
    agent_policy_loss_meter = AverageMeter()
    agent_value_loss_meter = AverageMeter()
    agent_entropy_loss_meter = AverageMeter()

    ## -- meter of time 
    interactive_gpu_time_meter = AverageMeter()
    wm_update_gpu_time_meter = AverageMeter()
    agent_update_gpu_time_meter = AverageMeter()
    interactive_wall_time_meter = AverageMeter()
    wm_update_wall_time_meter = AverageMeter()
    agent_update_wall_time_meter = AverageMeter()

    # -- init environment
    vec_env = env_factory.build_single_env(
        env_params, frame_skip=frame_skip, maxpooling=maxpooling) # TODO multiple env
    action_dims = list(vec_env.action_space.nvec)
    ActionParser.init(action_dims)


    # -- init replay buffer
    image_size = env_params["Environment"]["task_parameter"]["image_size"]
    
    replay_buffer = init_replay_buffer(
        device=device,
        obs_h=image_size[0], obs_w=image_size[1], obs_c=IMAGE_CHANNEL, 
        action_dims=action_dims, 
        num_envs=num_envs, 
        max_length=max_length, 
        warmup_length=warmup_length, 
        frame_skip=frame_skip,
        store_on_gpu=store_on_gpu,
    )

    if export_data_path:
        replay_buffer.load_trajectory(
            path=export_data_path
        )
    
    if pre_collected_data_path:
        replay_buffer.load_buffer(
            path=pre_collected_data_path
        )
    
    # -- init world model
    world_model = init_world_model(
        device=device,
        video_model_params=video_model_params,
        latent_action_enc_params=latent_action_enc_params,
        state_decoder_params=state_decoder_params,
        action_projector_params=action_projector_params,
        optimizer_params=optimizer_params,
        tensorlogger=tb_logger,
        pretrained_model_path=pretrained_model_path,
        fine_tune=fine_tune,
        use_amp=mixed_precision,
        amp_dtype=dtype
    )

    # -- init Agent
    agent = init_agent(
        feat_len=feat_len, 
        feat_dim=feat_dim, 
        num_layers=num_layers,
        hidden_dim=hidden_dim, 
        action_dim=action_dim, 
        gamma=gamma, 
        lambd=lambd, 
        entropy_coef=entropy_coef
    )

    # -- load training checkpoint
    if load_model or os.path.exists(latest_path):
        CheckpointIO.load(latest_path)

    # -- init context qeue
    num_frames = video_model_params["num_frames"]
    context_obs = deque(maxlen=num_frames)
    # context_action = deque(maxlen=num_frames)
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    
    # -- Training loop
    for total_steps in range(max_steps//num_envs):
        
        # >> training script
        def interactive():
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_obs) != num_frames:
                    action = vec_env.action_space.sample()
                else:
                    context_latent = world_model.encode(torch.cat(list(context_obs), dim=1))
                    action = agent.sample_as_env_action(
                        context_latent, greedy=False
                    )
                    action = np.squeeze(action)
            obs, reward, done, truncated, info = vec_env.step(action)
            context_obs.append(rearrange(torch.Tensor(current_obs.copy()).cuda(), "C H W -> 1 1 C H W")/255) # [one env , len obs ,(obs) ]
            # context_action.append(action) # STORM like actions buffer

            replay_buffer.append(current_obs, action, reward, done)
            sum_reward += reward
            current_obs = obs
            current_info = info

            truncated = np.array([truncated])
            done = np.array([done])
            done_flag = np.logical_or(done, truncated)
            if done_flag.any() :
                for i in range(num_envs):
                    if done_flag:
                        tb_logger.log(f"env/reward", sum_reward[i],type='scalar')
                        tb_logger.log(f"env/episode_steps", current_info["elapsed_steps"]//4,type='scalar')
                        tb_logger.log("replay_buffer/length", len(replay_buffer),type='scalar')
                        sum_reward[i] = 0
                        context_obs.clear()
                        current_obs, current_info = vec_env.reset()
            return None

        def world_model_train_step():
            world_model.train()
            agent.eval()

            obs, action, reward, termination  = replay_buffer.sample(
                batch_size=batch_size, external_batch_size=demon_batch_size, batch_length=seq_length, to_device=device
            )
            return world_model.update(
                sample_obs=obs, sample_action=action,
                sample_rewards=reward, sample_termin=termination,
            )
            

        def agent_train_step():
            obs, action, _, _  = replay_buffer.sample(
                batch_size=imagination_batch_size, 
                external_batch_size=imagination_demon_batch_size, 
                batch_length=imagination_context_length, to_device=device
            )

            context_latent = world_model.reset(
                sample_obs=obs, sample_action=action, buffer_size=imagination_seq_length)
            for i in range(imagination_seq_length):
                action = agent.sample(context_latent)
                context_latent = world_model.step(action)
            latents, actions, rewards, terminations = world_model.export()

            return agent.update(
                latent=latents,
                action=actions,
                old_logprob=None, # not use
                old_value=None, # not use
                reward=rewards,
                termination=terminations,
            )

            
        # >> Training flow
        # >> World model and Agent interactive with Env
        step_start_time = time.time()
        (interactive_debug_data), interactive_gpu_etime_ms= gpu_timer(interactive)
        interactive_wall_time_meter.update((time.time() - step_start_time) * 1000.)
        interactive_gpu_time_meter.update(interactive_gpu_etime_ms)
        if (total_steps % log_freq == 0):
            logger.info(
                f"(interactive)[{total_steps}] "
                f"[mem: {torch.cuda.max_memory_allocated()/1024.0**2:.2e}] "
                f"[gpu_time: {interactive_gpu_time_meter.avg:.1f} ms]"
                f"[wall_time: {interactive_wall_time_meter.avg:.1f} ms]"
            )

        # >> World model training
        if replay_buffer.ready() and total_steps % (world_model_interval//num_envs) == 0:
            step_start_time = time.time()
            (world_model_debug_data), world_model_train_gpu_etime_ms= gpu_timer(world_model_train_step)
            wm_update_wall_time_meter.update((time.time() - step_start_time) * 1000.)
            wm_update_gpu_time_meter.update(world_model_train_gpu_etime_ms)

            # >> World model Logging data
            total_loss = world_model_debug_data["loss"]["total"]
            action_loss = world_model_debug_data["loss"]["action"]
            reward_loss = world_model_debug_data["loss"]["reward"]
            termin_loss = world_model_debug_data["loss"]["termin"]

            optim_stats = world_model_debug_data["optim_stats"]
            new_lr = world_model_debug_data["lr"]
            new_wd = world_model_debug_data["wd"]

            wm_total_loss_meter.update(total_loss)
            wm_action_loss_meter.update(action_loss)
            wm_reward_loss_meter.update(reward_loss)
            wm_termin_loss_meter.update(termin_loss)
            
            # >> TB logger
            # TODO: Decoder decode JEPA feature or somtthing need record data

            # >> Print
            if (total_steps % log_freq == 0) or np.isnan(total_loss) or np.isinf(total_loss):
                logger.info(
                    f"(world_model_train_step)[{total_steps}] "
                    f"loss: {wm_total_loss_meter.avg:.3f} | "
                    f"act:{wm_action_loss_meter.avg:.3f} rew:{wm_reward_loss_meter.avg:.3f} ter:{wm_termin_loss_meter.avg:.3f} | "
                    f"[wd: {new_wd:.2e}] [lr: {new_lr:.2e}] "
                    f"[mem: {torch.cuda.max_memory_allocated()/1024.0**2:.2e}] "
                    f"[gpu_time: {wm_update_gpu_time_meter.avg:.1f} ms]"
                    f"[wall_time: {wm_update_wall_time_meter.avg:.1f} ms]"
                )

                if optim_stats is not None:
                    logger.info(
                        f"(world_model_train_step)[{total_steps}] "
                        f"first moment: {optim_stats.get('exp_avg').avg:.2e} [{optim_stats.get('exp_avg').min:2e} {optim_stats.get('exp_avg').max:2e}] "
                        f"second moment: {optim_stats.get('exp_avg_sq').avg:2e} [{optim_stats.get('exp_avg_sq').min:2e} {optim_stats.get('exp_avg_sq').max:2e}]")


                for name, value in world_model_debug_data["train_model_state"].items():
                   logger.info(f"(world_model_train_step)[{total_steps}] "
                               f"[{name}]: f/l[{value.first_layer:2e} {value.last_layer:2e}] "
                               f"mn/mx({value.min:2e}, {value.max:2e}) {value.global_norm:2e}")
             
                assert not np.isnan(total_loss), 'loss is nan'

        # >> Agent training
        if replay_buffer.ready() and total_steps % (agent_interval//num_envs) == 0:
            step_start_time = time.time()
            (agent_debug_data), agent_train_gpu_etime_ms= gpu_timer(agent_train_step)
            agent_update_wall_time_meter.update((time.time() - step_start_time) * 1000.)
            agent_update_gpu_time_meter.update(agent_train_gpu_etime_ms)

            # >> Agent Logging data
            total_loss = agent_debug_data["loss"]["total"]
            policy_loss = agent_debug_data["loss"]["policy"]
            value_loss = agent_debug_data["loss"]["value"]
            entropy_loss = agent_debug_data["loss"]["entropy"]

            optim_stats = agent_debug_data["optim_stats"]

            agent_total_loss_meter.update(total_loss)
            agent_policy_loss_meter.update(policy_loss)
            agent_value_loss_meter.update(value_loss)
            agent_entropy_loss_meter.update(entropy_loss)
            
            # >> Print
            if (total_steps % log_freq == 0) or np.isnan(total_loss) or np.isinf(total_loss):
                logger.info(
                    f"(world_model_train_step)[{total_steps}] "
                    f"loss: {agent_total_loss_meter.avg:.3f} | "
                    f"policy:{agent_policy_loss_meter.avg:.3f} value:{agent_value_loss_meter.avg:.3f} entropy:{agent_entropy_loss_meter.avg:.3f} | "
                    f"[mem: {torch.cuda.max_memory_allocated()/1024.0**2:.2e}] "
                    f"[gpu_time: {agent_update_gpu_time_meter.avg:.1f} ms]"
                    f"[wall_time: {agent_update_wall_time_meter.avg:.1f} ms]"
                )

                if optim_stats is not None:
                    logger.info(
                        f"(world_model_train_step)[{total_steps}] "
                        f"first moment: {optim_stats.get('exp_avg').avg:.2e} [{optim_stats.get('exp_avg').min:2e} {optim_stats.get('exp_avg').max:2e}] "
                        f"second moment: {optim_stats.get('exp_avg_sq').avg:2e} [{optim_stats.get('exp_avg_sq').min:2e} {optim_stats.get('exp_avg_sq').max:2e}]")


                for name, value in agent_debug_data["train_model_state"].items():
                   logger.info(f"(world_model_train_step)[{total_steps}] "
                               f"[{name}]: f/l[{value.first_layer:2e} {value.last_layer:2e}] "
                               f"mn/mx({value.min:2e}, {value.max:2e}) {value.global_norm:2e}")
             
                assert not np.isnan(total_loss), 'loss is nan'

        # >> Checkpoint save
        if total_steps % checkpoint_freq == 0:
            CheckpointIO.save(
            world=world_model, agent=agent,
            path=latest_path, logger=logger)
        if save_every_freq > 0 and total_steps % save_every_freq == 0:
            save_every_file = f'{tag}-step{total_steps}.pth.tar'
            save_every_path = os.path.join(folder, save_every_file)
            CheckpointIO.save(
                world=world_model, agent=agent,
                path=save_every_path, logger=logger)
            

    logger.info("Training Done!!!!!!1")

