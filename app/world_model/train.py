import torch
import numpy as np

from einops import rearrange
from libs import env_wrapper
from tqdm import tqdm
from collections import deque
from app.world_model import utils
from src.models.agents.agents import ActorCriticAgent
from app.world_model.replay_buffer import ReplayBuffer
from src.models.world_models.base_world_model import WorldModelBase

#Tensorboard Logger
logger = utils.Logger()

def train_world_model_step(
        world_model:WorldModelBase,
        replay_buffer:ReplayBuffer,
        batch_size,
        batch_length,
        demonstration_batch_size,
        logger:utils.Logger,
        log_video,
        **kwargs
    ):
    obs, action, reward, termination = replay_buffer.sample(batch_size, demonstration_batch_size, batch_length)
    world_model.update(obs, action, reward, termination, logger=logger, log_video=log_video,**kwargs)

def world_model_imagine_data(
        world_model:WorldModelBase,
        replay_buffer:ReplayBuffer,
        agent:ActorCriticAgent,
        imagine_batch_size,
        imagine_batch_length,
        imagine_context_length,
        imagine_demonstration_batch_size,
        log_video,
        logger:utils.Logger,
    ):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()
    obs, action, _, _ = replay_buffer.sample(imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length)
    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent=agent, sample_obs=obs, sample_action=action,
        imagine_batch_size=imagine_batch_size+imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger
    )
    return latent, action, reward_hat, termination_hat


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
    maxpooling = True
    
    # >>> Create env
    vec_env = env_wrapper.build_single_env(args,frame_skip=frame_skip, maxpooling=maxpooling) # TODO multiple env
    action_dims = list(vec_env.action_space.nvec)
    
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
    current_obs, current_info = vec_env.reset()

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

    # >>> Sample and Training 
    for total_steps in tqdm(range(max_steps//num_envs)):
        #  >>> sample part
        if replay_buffer.ready():
            # world_model.eval()
            # agent.eval()
            # with torch.no_grad():
            #     # emb_code = world model encode 
            #     # actio = agent.sample(emb_code, greedy=False)
            action = vec_env.action_space.sample()
        else:
            action = vec_env.action_space.sample()
            
        # current_obs shape convert to [a env, obs len, (obs)] the obs len will affect for maxpooling and frame skip
        context_obs.append(rearrange(torch.Tensor(current_obs.copy()).cuda(), "C H W -> 1 1 C H W")/255) 
        context_action.append(action)
        obs, reward, done, truncated, info = vec_env.step(action)
        replay_buffer.append(current_obs, action, reward, done)

        # update current status 
        sum_reward += reward
        current_obs = obs
        current_info = info

        truncated = np.array([truncated])
        done = np.array([done])
        done_flag = np.logical_or(done, truncated)
        if done_flag.any() :
            for i in range(num_envs):
                if done_flag:
                    logger.log(f"sample/{env_name}_reward", sum_reward[i])
                    logger.log(f"sample/{env_name}_episode_steps", current_info["elapsed_steps"]//frame_skip)
                    logger.log("replay_buffer/length", len(replay_buffer))
                    sum_reward[i] = 0
                    current_obs, current_info = vec_env.reset()

        # >>> train world model part
        if replay_buffer.ready() and (total_steps < 2500 or total_steps % (train_dynamics_every_steps//num_envs) == 0):
            log_video = total_steps % (save_every_steps//num_envs) == 0
            train_world_model_step(
                world_model=world_model,
                replay_buffer=replay_buffer,
                batch_size=batch_size,
                batch_length=batch_length,
                demonstration_batch_size=demonstration_batch_size,
                logger=logger,
                log_video=log_video,
                step=total_steps,
                max_steps=max_steps
            )


        # >>> train agent part
        if False and replay_buffer.ready() and total_steps % (train_agent_every_steps//num_envs) == 0 and total_steps*num_envs >= 0:
            if total_steps % (save_every_steps//num_envs) == 0:
                log_video = True
            else:
                log_video = False

            imagine_latent, \
            agent_action, agent_logprob, agent_value, \
            imagine_reward, imagine_termination \
            = world_model_imagine_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                imagine_batch_size=imagine_batch_size,
                imagine_batch_length=imagine_batch_length,
                imagine_context_length=imagine_context_length,
                imagine_demonstration_batch_size=imagine_demonstration_batch_size,
                log_video=log_video,
                logger=logger
            )

            agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logprob=agent_logprob,
                old_value=agent_value,
                reward=imagine_reward,
                termination=imagine_termination,
                logger=logger
            )

        # >>> log and save model
        # Evaluate model and save best model
        if total_steps % (eval_every_steps//num_envs) == 0 and total_steps > 0:
            # rewards, steps = eval mdoel
            # if reward > max_reward or steps < min_steps
            #   save model
            pass
        # save model per episode
        if total_steps % (save_every_steps//num_envs) == 0:
            # save model
            pass
    logger.close()
    vec_env.close()