import torch
import numpy as np
import os
import shutil
import yaml

from einops import rearrange
from libs import env_wrapper
from tqdm import tqdm
from collections import deque
from app.world_model import utils
from app.world_model.replay_buffer import ReplayBuffer
from src.models.agents.agents import ActorCriticAgent
from src.models.world_models.jepa_world_model import JEPAWorldModel

#Tensorboard Logger
tensorboard_logger = utils.Logger()

def imagine_data(world_model:JEPAWorldModel,agent: ActorCriticAgent, 
                sample_obs, sample_action,
                imagine_batch_size, imagine_batch_length, clip_len,
                logger, 
                log_video):
    world_model.eval()
    agent.eval()
    # initial buffer
    B, C, T, H, W = sample_obs.shape
    T = T//world_model.tubelet_size # image space len to latent space len
    latent_size = (imagine_batch_size, T + imagine_batch_length, world_model.get_num_patches(), world_model.embed_dim)
    action_size = (imagine_batch_size, T + imagine_batch_length, len(world_model.action_dims))
    scalar_size = (imagine_batch_size, T + imagine_batch_length)

    embedding_buffer = torch.zeros(latent_size, dtype=world_model.dtype, device="cuda")
    action_buffer = torch.zeros(action_size, dtype=world_model.dtype, device="cuda")
    reward_hat_buffer = torch.zeros(scalar_size, dtype=world_model.dtype, device="cuda")
    termination_hat_buffer = torch.zeros(scalar_size, dtype=world_model.dtype, device="cuda")

    # Encode smaple observation
    embedding = world_model.encode_obs(sample_obs) # B,C,T,H,W -> B,(T P), D
    embedding = rearrange(embedding, "B (T P) D -> B T P D",B=B, T=T,D=world_model.embed_dim)
    embedding_buffer[:, :T] = embedding
    action_buffer[:,:T] = sample_action[:, ::world_model.tubelet_size]
    reward_hat_buffer[:,:T] = 0 # 0 -> T is dummy 
    termination_hat_buffer[:,:T] = 0 # 0 -> T is dummy

    for i in range(imagine_batch_length):
        #repeat and save data
        # 1. Get current context embedding
        predict_embedding = rearrange(
            embedding_buffer[:, i:i+T], "B T P D -> B (T P) D"
        )  
        interactive_embedding = embedding_buffer[:, i+T-agent.feat_len:i+T]

        # 2. Actor use prediect latent sample action
        with torch.no_grad():
            # pred_embedding shape: [B, P, D] → flatten or pooled
            action = agent.sample(interactive_embedding).unsqueeze(1)  # ➜ [B, 1, A]
            
        # 3. predict next frame embedding + reward/termination
        current_actions = torch.cat([action_buffer[:, i:i+clip_len], action],dim=1) # [B, T, A]
        with torch.no_grad():
            completed_embedding, pred_embedding, reward_hat, termination_hat = \
                world_model.step(predict_embedding, current_actions)
            
        # 4. update buffer
        embedding_buffer[:, i+T:i+T+1] = pred_embedding.unsqueeze(1)        # [B, 1, P, D]
        action_buffer[:, i+T:i+T+1] = action                                # [B, 1, A]
        reward_hat_buffer[:, i+T:i+T+1] = reward_hat.unsqueeze(1)           # [B, 1]
        termination_hat_buffer[:, i+T:i+T+1] = termination_hat.unsqueeze(1) # [B, 1]

    return  embedding_buffer[:,T-agent.feat_len:], \
            action_buffer[:,T:], \
            reward_hat_buffer[:,T:], \
            termination_hat_buffer[:,T:]

def train_world_model_step(
        world_model:JEPAWorldModel,
        replay_buffer:ReplayBuffer,
        batch_size,
        batch_length,
        demonstration_batch_size,
        logger:utils.Logger,
        log_video,
        **kwargs
    ):
    if replay_buffer.ready():
        obs, action, reward, termination = replay_buffer.sample(batch_size, demonstration_batch_size, batch_length)
        world_model.update(obs, action, reward, termination, logger=logger, log_video=log_video,**kwargs)

def world_model_imagine_data(
        world_model:JEPAWorldModel,
        replay_buffer:ReplayBuffer,
        agent:ActorCriticAgent,
        imagine_batch_size,
        imagine_batch_length,
        imagine_context_length,
        imagine_demonstration_batch_size,
        clip_len,
        log_video,
        logger:utils.Logger,
    ):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()
    obs, action, _, _ = replay_buffer.sample(imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length)
    if log_video:
        visual_obs = rearrange(obs, "B C T H W ->B T C H W")
        logger.log("Imagine/agent_sample_video", torch.clamp(visual_obs[::imagine_batch_size//16], 0, 1).cpu().float().detach().numpy())

    return imagine_data(
        world_model=world_model ,agent=agent, sample_obs=obs, sample_action=action,
        imagine_batch_size=imagine_batch_size+imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length, clip_len=clip_len,
        log_video=log_video,
        logger=logger
    )

mount_path_env = os.getenv('MOUNT_PATH', "")
def main(args, resume_preempt=False):
    # ## Test
    # from app.world_model.unittest import main
    # main(args, resume_preempt)
    # return
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
    logger_path = logger_setting.get("folder")
    log_save_path = os.path.join(mount_path_env, f"runs/{logger_path}")
    dummy_config_path = os.path.join(mount_path_env, f"runs/{logger_path}/config.yaml")
    ckpt_path = os.path.join(mount_path_env, f"ckpt/{logger_path}")

    env_name = env_setting.get("task")
    tensorboard_logger.init(log_save_path)
    num_envs = joint_train_agent.get("NumEnvs")
    frame_skip = args["Models"]["WorldModel"]["tubelet_size"]
    maxpooling = True

    # >>> dump config file and create log/ckpt dir 
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(logger_path, exist_ok=True)
    with open(dummy_config_path, "w") as f:
        yaml.dump(args, f, default_flow_style=False)

    # >>> Create env 
    vec_env = env_wrapper.build_single_env(args,frame_skip=frame_skip, maxpooling=maxpooling) # TODO multiple env
    action_dims = list(vec_env.action_space.nvec)

    # >>> Build up replay buffer
    replay_buffer = utils.build_replay_buffer(
        args, action_dims, 
        device=device if basic_setting.get("ReplayBufferOnGPU") else "cpu")
    # replay_buffer.load_buffer("/home/cgv/Documents/project/EmbodiedAgent/v-jepa/test_1024.npz")

    if joint_train_agent.get("UseDemonstration"):
        path = joint_train_agent.get("DemonstrationPath")
        print(f"Loading demonstration trajectory from {path}")
        # utils.logger.info(f"Loading demonstration trajectory from {path}")
        replay_buffer.load_trajectory(path=path)

    # >>> Build up model
    world_model = utils.build_world_model(args, action_dims, device=device)
    agent = utils.build_agent(args, action_dims, device=device)
    
    # initial variable
    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()


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

    # init context qeue
    context_obs = deque(maxlen=batch_length)
    context_action = deque(maxlen=batch_length)

    clip_len = imagine_context_length // world_model.tubelet_size
    print("start training")
    # >>> Sample and Training 
    for total_steps in tqdm(range(max_steps//num_envs)):
        #  >>> sample part
        if replay_buffer.ready() and len(context_obs) == batch_length:
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                embedding = world_model.encode_obs(torch.cat(list(context_obs), dim=2).to(device=device)) # B,C,T,H,W -> B,(T P), D
                embedding = rearrange(embedding, "B (T P) D -> B T P D",T=batch_length//world_model.tubelet_size)[:, -agent.feat_len:]
                action = agent.sample(embedding, greedy=False) # ➜ [B, A]
                action = np.squeeze(action.cpu())
            # action = vec_env.action_space.sample()
        else:
            action = vec_env.action_space.sample()
            
        # current_obs shape convert to [a env, obs len, (obs)] the obs len will affect for maxpooling and frame skip
        last_obs, reward, done, truncated, info = vec_env.step(action)
        for _obs in current_info["all_obs"]:
            context_obs.append(rearrange(torch.from_numpy(_obs.copy()).float().cuda(), "C H W -> 1 C 1 H W") / 255)
            context_action.append(action)
            replay_buffer.append(_obs, action, reward, done)

        # update current status 
        sum_reward += reward
        current_obs = last_obs
        current_info = info

        truncated = np.array([truncated])
        done = np.array([done])
        done_flag = np.logical_or(done, truncated)
        if done_flag.any() :
            for i in range(num_envs):
                if done_flag:
                    tensorboard_logger.log(f"sample/{env_name}_reward", sum_reward[i])
                    tensorboard_logger.log(f"sample/{env_name}_episode_steps", current_info["elapsed_steps"]//frame_skip)
                    tensorboard_logger.log("replay_buffer/length", len(replay_buffer))
                    sum_reward[i] = 0
                    current_obs, current_info = vec_env.reset()

        # >>> train world model part
        if replay_buffer.ready() and (total_steps % (train_dynamics_every_steps//num_envs) == 0):
            log_video = total_steps % (save_every_steps//num_envs) == 0
            train_world_model_step(
                world_model=world_model,
                replay_buffer=replay_buffer,
                batch_size=batch_size,
                batch_length=batch_length,
                demonstration_batch_size=demonstration_batch_size,
                logger=tensorboard_logger,
                log_video=log_video,
                step=total_steps,
                max_steps=max_steps
            )



        # >>> train agent part
        if replay_buffer.ready() and total_steps % (train_agent_every_steps//num_envs) == 0 and total_steps*num_envs >= 0:
            if total_steps % (save_every_steps//num_envs) == 0:
                log_video = True
            else:
                log_video = False

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
                log_video=log_video,
                logger=tensorboard_logger
            )

            agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logprob=None,
                old_value=None,
                reward=imagine_reward,
                termination=imagine_termination,
                clip_len=clip_len,
                logger=tensorboard_logger
            )

        # >>> log and save model
        # Evaluate model and save best model
        # if total_steps % (eval_every_steps//num_envs) == 0 and total_steps > 0:
        #     # rewards, steps = eval mdoel
        #     # if reward > max_reward or steps < min_steps
        #     #   save model
        #     pass
        # save model per episode
        if total_steps % (save_every_steps//num_envs) == 0 and total_steps > 0:
            print(f"Saving model at total steps {total_steps}")
            # utils.logger.info(f"Saving model at total steps {total_steps}")
            torch.save(world_model.state_dict(), f"{ckpt_path}/world_model_{total_steps}.pth")
            torch.save(agent.state_dict(), f"{ckpt_path}/agent_{total_steps}.pth")
            replay_buffer.export_buffer(f"{env_name}_sampe")

    tensorboard_logger.close()
    vec_env.close()