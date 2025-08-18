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

from app.world_model.utils import (
    init_opt,
    init_video_model,
    init_latent_action_encoder,
    load_checkpoint,
    load_jepa_encoder,
    init_replay_buffer,
    init_world_model,
    init_agent
)

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__)

def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    cfgs_meta = args.get('meta')
    load_model = cfgs_meta.get('load_checkpoint') or resume_preempt
    r_file = cfgs_meta.get('read_checkpoint', None)
    seed = cfgs_meta.get('seed', _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get('save_every_freq', -1)
    skip_batches = cfgs_meta.get('skip_batches', -1)
    use_sdpa = cfgs_meta.get('use_sdpa', False)
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

    # -- MASK
    cfgs_mask = args.get('mask')

    # -- MODEL
    cfgs_model = args.get('model')
    model_name = cfgs_model.get('model_name')
    pred_depth = cfgs_model.get('pred_depth')
    pred_embed_dim = cfgs_model.get('pred_embed_dim')
    uniform_power = cfgs_model.get('uniform_power', True)
    use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
    zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)
    la_enc_num_heads = cfgs_model.get('latent_action_num_heads', 8)
    dims_aciton_codebook = cfgs_model.get('dims_aciton_codebook', 10)
    number_aciton_codebook = cfgs_model.get('number_aciton_codebook', 1)
    lfq_bias = cfgs_model.get('lfq_bias', True)
    lfq_commit_weight = cfgs_model.get('lfq_commit_weight', 0.25)
    lfq_entropy_weight = cfgs_model.get('lfq_entropy_weight', 0.1)
    lfq_diversity_weight = cfgs_model.get('lfq_diversity_weight', 1.0)
    training_model_list = cfgs_model.get('training_model_list', ["encoder", "predictor", "latent_action_enc"])
    action_adapter_type = cfgs_model.get('action_adapter_type', "None")

    # -- DATA
    cfgs_data = args.get('data')
    dataset_type = cfgs_data.get('dataset_type', 'videodataset')
    mask_type = cfgs_data.get('mask_type', 'multiblock3d')
    dataset_paths = cfgs_data.get('datasets', [])
    datasets_weights = cfgs_data.get('datasets_weights', None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), 'Must have one sampling weight specified for each dataset'
    batch_size = cfgs_data.get('batch_size')
    num_clips = cfgs_data.get('num_clips')
    num_frames = cfgs_data.get('num_frames')
    tubelet_size = cfgs_data.get('tubelet_size')
    sampling_rate = cfgs_data.get('sampling_rate')
    duration = cfgs_data.get('clip_duration', None)
    crop_size = cfgs_data.get('crop_size', 224)
    patch_size = cfgs_data.get('patch_size')
    pin_mem = cfgs_data.get('pin_mem', False)
    num_workers = cfgs_data.get('num_workers', 1)
    filter_short_videos = cfgs_data.get('filter_short_videos', False)
    decode_one_clip = cfgs_data.get('decode_one_clip', True)
    log_resource_util_data = cfgs_data.get('log_resource_utilization', False)

    # -- LOSS
    cfgs_loss = args.get('loss')
    loss_exp = cfgs_loss.get('loss_exp')
    reg_coeff = cfgs_loss.get('reg_coeff')
    quant_coeff = cfgs_loss.get('quant_coeff')

    # -- OPTIMIZATION
    cfgs_opt = args.get('optimization')
    ipe = cfgs_opt.get('ipe', None)
    ipe_scale = cfgs_opt.get('ipe_scale', 1.0)
    clip_grad = cfgs_opt.get('clip_grad', None)
    wd = float(cfgs_opt.get('weight_decay'))
    final_wd = float(cfgs_opt.get('final_weight_decay'))
    num_epochs = cfgs_opt.get('epochs')
    warmup = cfgs_opt.get('warmup')
    start_lr = cfgs_opt.get('start_lr')
    lr = cfgs_opt.get('lr')
    final_lr = cfgs_opt.get('final_lr')
    ema = cfgs_opt.get('ema')
    betas = cfgs_opt.get('betas', (0.9, 0.999))
    eps = cfgs_opt.get('eps', 1.e-8)

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
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- init logger
    logger.info('Initializing loader...')
    tensor_logger = TensorboardLogger(path=folder)
    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'loss'),
        ('%.5f', 'loss-jepa'),
        ('%.5f', 'loss-quant'),
        ('%.5f', 'reg-loss'),
        ('%.5f', 'enc-grad-norm'),
        ('%.5f', 'pred-grad-norm'),
        ('%.5f', 'la-grad-norm'),
        ('%d', 'gpu-time(ms)'),
        ('%d', 'wall-time(ms)'),
    )
    loss_meter = AverageMeter()
    input_var_meter = AverageMeter()
    input_var_min_meter = AverageMeter()
    jepa_loss_meter = AverageMeter()
    quant_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()
    mask_meters = [AverageMeter() for _ in range(len(cfgs_mask))]
    gpu_time_meter = AverageMeter()
    wall_time_meter = AverageMeter()

    # -- init environment
    vec_env = env_factory.build_single_env(args, frame_skip=frame_skip, maxpooling=maxpooling) # TODO multiple env
    action_dims = list(vec_env.action_space.nvec)

    # -- init replay buffer
    replay_buffer = init_replay_buffer(
        device=device,
        obs_h=image_size, obs_w=image_size, obs_c=3, 
        action_dims=action_dims, 
        num_envs=num_envs, 
        max_length=replay_buffer_max_len, 
        warmup_length=replay_buffer_warmup_len, 
        frame_skip=skip_frame,
        store_on_gpu=replay_buffer_store_on_gpu,
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
    world_model = init_world_model()

    # -- init Agent
    agent = init_agent()

    # -- load training checkpoint
    if load_model or os.path.exists(latest_path):
        (
            encoder,
            predictor,
            target_encoder,
            latent_action_enc,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            latent_action_enc=latent_action_enc,
            opt=optimizer,
            scaler=scaler)

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'latent_action_encoder': latent_action_enc.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f'Encountered exception when saving checkpoint: {e}')

    # -- Training loop
    for total_steps in range(max_steps//num_envs):
        itr_start_time = time.time()

        # >> training script
        def interactive():
            pass

        def world_model_train_step():
            _new_lr = scheduler.step()
            _new_wd = wd_scheduler.step()
            # --

        def agent_train_step():
            pass
            
        # >> training flow
        (debug_data), interactive_gpu_etime_ms= gpu_timer(interactive)
        (debug_data), world_model_train_gpu_etime_ms= gpu_timer(world_model_train_step)
        (debug_data), agent_train_gpu_etime_ms= gpu_timer(agent_train_step)

        # >> Logging data
        iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.
        loss_meter.update(loss)
        input_var = float(AllReduce.apply(clips.view(clips.shape[0], -1).var(dim=1).mean(dim=0)))
        input_var_min = float(AllReduce.apply(torch.min(clips.view(clips.shape[0], -1).var(dim=1))))
        input_var_meter.update(input_var)
        input_var_min_meter.update(input_var_min)
        jepa_loss_meter.update(loss_jepa)
        quant_loss_meter.update(loss_quant)
        reg_loss_meter.update(loss_reg)
        gpu_time_meter.update(gpu_etime_ms)
        wall_time_meter.update(iter_elapsed_time_ms)

        csv_logger.log(
            epoch + 1,
            itr,
            loss,
            loss_jepa,
            loss_quant,
            loss_reg,
            grad_stats.global_norm,
            grad_stats_pred.global_norm,
            grad_stats_la_enc.global_norm,
            gpu_etime_ms,
            iter_elapsed_time_ms)
        
        if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
            logger.info(
                '[%d, %5d] loss: %.3f | p:%.3f q:%.3f r:%.3f | '
                'input_var: %.3f %.3f | '
                'masks: %s '
                '[wd: %.2e] [lr: %.2e] '
                '[mem: %.2e] '
                '[gpu: %.1f ms]'
                '[wall: %.1f ms]'
                % (epoch + 1, itr,
                    loss_meter.avg,
                    jepa_loss_meter.avg,
                    quant_loss_meter.avg,
                    reg_loss_meter.avg,
                    input_var_meter.avg,
                    input_var_min_meter.avg,
                    '[' + ', '.join(['%.1f' % m.avg for m in mask_meters]) + ']',
                    _new_wd,
                    _new_lr,
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                    gpu_time_meter.avg,
                    wall_time_meter.avg))

            if optim_stats is not None:
                logger.info(
                    '[%d, %5d] first moment: %.2e [%.2e %.2e] second moment: %.2e [%.2e %.2e]'
                    % (epoch + 1, itr,
                        optim_stats.get('exp_avg').avg,
                        optim_stats.get('exp_avg').min,
                        optim_stats.get('exp_avg').max,
                        optim_stats.get('exp_avg_sq').avg,
                        optim_stats.get('exp_avg_sq').min,
                        optim_stats.get('exp_avg_sq').max))

            if world_grad_stats is not None:
                logger.info(
                    '[%d, %5d] enc_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                    % (epoch + 1, itr,
                        grad_stats.first_layer,
                        grad_stats.last_layer,
                        grad_stats.min,
                        grad_stats.max,
                        grad_stats.global_norm))

                    
            assert not np.isnan(loss), 'loss is nan'

    # -- Save Checkpoint
    logger.info('avg. loss %.3f' % loss_meter.avg)
    # -- Save Last
    if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
        save_checkpoint(epoch + 1, latest_path)
        if save_every_freq > 0 and epoch % save_every_freq == 0:
            save_every_file = f'{tag}-e{epoch}.pth.tar'
            save_every_path = os.path.join(folder, save_every_file)
            save_checkpoint(epoch + 1, save_every_path)
