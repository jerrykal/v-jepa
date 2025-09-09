import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
from libs.environment_factory import build_single_env
from app.world_model.utils import init_world_model, init_replay_buffer
from src.utils.action_parser import ActionParser

# From: data.* and model.* in your YAML
video_model_params = {
    "patch_size": 16,             # Size of each visual patch (data.patch_size)
    "num_frames": 16,             # Number of frames per clip (data.num_frames)
    "tubelet_size": 2,            # Frames per tubelet (data.tubelet_size)
    "model_name": "vit_huge",     # Backbone model (model.model_name)
    "crop_size": 224,             # Spatial crop size (data.crop_size)
    "pred_depth": 12,             # Predictor depth (model.pred_depth)
    "pred_embed_dim": 384,        # Predictor embedding dim (model.pred_embed_dim)
    "uniform_power": True,        # Use uniform masking power (model.uniform_power)
    "use_mask_tokens": True,      # Whether to use mask tokens (model.use_mask_tokens)
    "adapter_type": "Concat",     # Action adapter type (model.action_adapter_type)
    "use_sdpa": False,             # Whether to enable SDPA (meta.use_sdpa)
}

# From: model.* (LFQ/VQ settings). YAML has typos "aciton"—mapped correctly below.
latent_action_enc_params = {
    "num_heads": 8,               # Num attention heads (model.latent_action_num_heads)
    "d_codebook": 8,              # Codebook vector dim (model.dims_aciton_codebook)
    "n_codebook": 2,              # Number of codebooks (model.number_aciton_codebook)
    "vq_bias": True,              # Use bias in quantizer (model.lfq_bias)
    "vq_commit_weight": 0.25,     # Commitment loss weight (model.lfq_commit_weight)
    "vq_entropy_weight": 0.1,     # Entropy loss weight (model.lfq_entropy_weight)
    "vq_diversity_weight": 1.0,   # Diversity loss weight (model.lfq_diversity_weight)
}

state_decoder_params = {
    # >> pooler
    "pooler_num_queries": 1,      # Number of learnable queries for pooling
    "pooler_num_heads": 4,        # Number of attention heads in pooling
    "mlp_ratio": 4.0,             # MLP hidden size multiplier
    "pooler_depth": 2,            # Number of transformer blocks in pooler
    "norm_layer": "LayerNorm",    # Normalization type ("LayerNorm", "BatchNorm", etc.)
    "init_std": 0.02,              # Weight initialization standard deviation
    "qkv_bias": True,             # Whether to use bias in QKV projections
    "complete_block": False,      # Whether to use a complete transformer block

    # >> rewards decoder
    "reward_hidden_dim": 256,     # Hidden dimension for reward decoder
    "reward_depth": 2,            # Depth (number of layers) of reward decoder
    "reward_num_classes": 255,      # Output dimension (1 for regression)

    # >> termination decoder
    "termin_hidden_dim": 256,     # Hidden dimension for termination decoder
    "termin_depth": 2             # Depth of termination decoder
}

action_projector_params = {
    "hidden_dims": 256,    # Hidden dimensions for action projector MLP
    "depth": 2                    # Number of layers in action projector
}

optimizer_params = {
    "start_lr": 1e-4,             # Initial learning rate
    "ref_lr": 1e-3,               # Reference (peak) learning rate
    "warmup_ratio": 0.1,          # Warmup steps ratio relative to total steps
    "wd": 0.05,                   # Weight decay
    "final_wd": 0.0,              # Final weight decay after scheduling
    "final_lr": 1e-5,             # Final learning rate after scheduling
    "mixed_precision": True,      # Whether to use mixed precision training
    "total_steps": 100000,        # Total number of training steps
    "betas": (0.9, 0.95),         # Adam optimizer betas
    "eps": 1e-8,                  # Adam optimizer epsilon
    "zero_init_bias_wd": False    # Apply weight decay to zero-initialized biases
}

pretrained_model_path = "/media/cgv/1tb_disk/download/ac_jepa-latest.pth.tar"
fine_tune = False
tensor_logger = None  # TensorboardLogger(path="logs/")


# env params
params = {
    "Environment":{
        "task": "CombatSpider",
        "task_parameter": {
            "image_size": [224, 224],
            "step_penalty": 0,
            "attack_reward": 1,
            "success_reward": 10,
            # "max_spawn_range": 10,
            # "target_quantities": 1,
            # "max_episode_len": 500,
        },
        "seed": 123,
        "action_space": "ReducedActionSpace",
        "observation": "DefaultObservation" 

    }
}
dummy_env = build_single_env(params)
action_dims = list(dummy_env.action_space.nvec)
ActionParser.init(action_dims)

replay_buffer = init_replay_buffer(
    "cpu",
    obs_h=224, obs_w=224, obs_c=3, action_dims=action_dims, num_envs=1, 
    max_length=int(1E5), warmup_length=50000, frame_skip=4,
    store_on_gpu=False,
)
replay_buffer.load_buffer("/home/cgv/Documents/project/EmbodiedAgent/v-jepa/test_1024.npz")

wm = init_world_model(
    device="cuda",
    video_model_params=video_model_params,
    latent_action_enc_params=latent_action_enc_params,
    state_decoder_params=state_decoder_params,
    action_projector_params=action_projector_params,
    optimizer_params=optimizer_params,
    tensorlogger=tensor_logger,
    action_dims=action_dims,
    pretrained_model_path=pretrained_model_path,
    fine_tune=fine_tune,
    use_amp=False,
    amp_dtype=torch.float16
)

obs, action, reward, termination  = replay_buffer.sample(batch_size=16, external_batch_size=0, batch_length=16, to_device="cuda")
debug_result = wm.update(
    sample_obs=obs, sample_action=action,
    sample_rewards=reward, sample_termin=termination,
)

for name, value in debug_result["train_model_state"].items():
    print('[%s]: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
            % (name,
                value.first_layer,
                value.last_layer,
                value.min,
                value.max,
                value.global_norm))
    
optim_stats = debug_result["optim_stats"]
print("[%s]first moment: %.2e [%.2e %.2e] second moment: %.2e [%.2e %.2e]"                      
        % ("optim_stats",
            optim_stats.get('exp_avg').avg,
            optim_stats.get('exp_avg').min,
            optim_stats.get('exp_avg').max,
            optim_stats.get('exp_avg_sq').avg,
            optim_stats.get('exp_avg_sq').min,
            optim_stats.get('exp_avg_sq').max))
                            
