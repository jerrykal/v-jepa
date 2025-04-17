import torch
import numpy as np

from einops import rearrange
from libs import env_wrapper
from app.world_model.train import world_model_imagine_data
from collections import deque
from app.world_model import utils
from src.models.agents.agents import ActorCriticAgent
from app.world_model.replay_buffer import ReplayBuffer
from src.models.world_models.jepa_world_model import JEPAWorldModel

#Tensorboard Logger
logger = utils.Logger()


def main(args, resume_preempt=False):
    # >>> Set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # >>> Get parameter
    joint_train_agent = args.get("JointTrainAgent")
    env_setting = args.get("Environment")
    logger_setting = args.get("logging")
    basic_setting = args.get("BasicSettings")

    
    env_name = env_setting.get("task")
    logger.init(logger_setting.get("folder"))
    num_envs = joint_train_agent.get("NumEnvs")
    frame_skip = 4

    action_dims=[12,3]
    
    # >>> Build up replay buffer
    replay_buffer = utils.build_replay_buffer(
        args, action_dims, 
        device=device if basic_setting.get("ReplayBufferOnGPU") else "cpu")
    
    if joint_train_agent.get("UseDemonstration"):
        path = joint_train_agent.get("DemonstrationPath")
        utils.logger.info(f"Loading demonstration trajectory from {path}")
        replay_buffer.load_trajectory(path=path)

    # >>> Build up model
    world_model = utils.build_world_model(args, action_dims, device=device)
    agent = utils.build_agent(args, action_dims, device=device)
    
    # initial variable
    # reset envs and variables
    sum_reward = np.zeros(num_envs)


    # init context qeue
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    # trainin setting
    max_steps                           = joint_train_agent.get("SampleMaxSteps")
    save_every_steps                    = joint_train_agent.get("SaveEverySteps")
    eval_every_steps                    = joint_train_agent.get("EvalEverySteps")
    train_agent_every_steps             = joint_train_agent.get("TrainAgentEverySteps")
    train_dynamics_every_steps          = joint_train_agent.get("TrainDynamicsEverySteps")
    batch_size                          = joint_train_agent.get("BatchSize")
    batch_length                        = joint_train_agent.get("BatchLength")
    demonstration_batch_size            = joint_train_agent.get("DemonstrationBatchSize") if joint_train_agent.get("UseDemonstration") else 0
    imagine_batch_size                  = joint_train_agent.get("ImagineBatchSize")
    imagine_batch_length                = joint_train_agent.get("ImagineBatchLength")
    imagine_context_length              = joint_train_agent.get("ImagineContextLength")
    imagine_demonstration_batch_size    = joint_train_agent.get("ImagineDemonstrationBatchSize") if joint_train_agent.get("UseDemonstration") else 0


    replay_buffer.load_buffer("test.pkl")
    clip_len=2
    imagine_latent, \
    agent_action, \
    imagine_reward, imagine_termination \
    = world_model_imagine_data(
        replay_buffer=replay_buffer,
        world_model=world_model,
        agent=agent,
        imagine_batch_size=imagine_batch_size,
        imagine_batch_length=imagine_batch_length,
        imagine_context_length=imagine_context_length,
        imagine_demonstration_batch_size=imagine_demonstration_batch_size,
        clip_len=clip_len,
        log_video=False,
        logger=logger
    )

    agent.update(
        latent=imagine_latent,
        action=agent_action,
        old_logprob=None,
        old_value=None,
        reward=imagine_reward,
        termination=imagine_termination,
        clip_len=clip_len,
        logger=logger
    )


    logger.close()
