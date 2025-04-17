import gym
import gymnasium
import numpy as np
from collections import deque

# MineDojo
from libs.mine_env import build_env

def build_single_env(params, frame_skip=4, maxpooling=True, seed = None)->gymnasium.Wrapper:
    env = build_env(params, seed)
    env = MineDojoGymnasium(
        minedojo_env=env,
        skip=frame_skip,
        seed=seed,
        maxpooling=maxpooling
    )
    return env

# MineDojo Gymnasium Wrapper
class MineDojoGymnasium(gymnasium.Env):
    def __init__(self, minedojo_env:gym.Wrapper, seed, skip=4, maxpooling=True):
        super().__init__()
        self.minedojo_env = minedojo_env
        self.skip = skip
        self._seed = seed
        self._maxpooling = maxpooling

        self.observation_space = minedojo_env.observation_space["rgb"]
        self.action_space = minedojo_env.action_space

        self.obs_buffer = deque(maxlen=2)
        self._elapsed_steps = 0

    def step(self, action):
        all_obs = []
        self.obs_buffer.clear()
        total_reward = 0

        for _ in range(self.skip):
            obs, reward, done, info = self.minedojo_env.step(action)
            self._elapsed_steps += 1
            self.obs_buffer.append(obs['rgb'])
            all_obs.append(obs['rgb'])
            
            total_reward += reward
            if done:
                break
        if len(self.obs_buffer) == 1:
            obs_image = self.obs_buffer[0] 
        else:
            obs_image = np.max(np.stack(self.obs_buffer), axis=0) if self._maxpooling else np.stack(self.obs_buffer)
        
        truncated = False
        return obs_image, total_reward, done, truncated, \
                {
                    'elapsed_steps':self._elapsed_steps,
                    'all_obs':all_obs
                }

    def reset(self, **kwargs):
        obs = self.minedojo_env.reset()
        self._elapsed_steps = 0
        self.minedojo_env.seed(self._seed)
        return obs['rgb'],\
            {
                'elapsed_steps':1,
                'all_obs':[obs['rgb']]
            }

    def close(self):
        self.minedojo_env.close()

    def render(self, mode='human'):
        return self.minedojo_env.render(mode=mode)
    