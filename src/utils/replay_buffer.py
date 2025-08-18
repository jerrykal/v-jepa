import numpy as np
import torch
from einops import rearrange
import pickle

class ReplayBuffer():
    def __init__(self, 
                 obs_shape, action_dims, 
                 num_envs, 
                 max_length=int(1E5), warmup_length=50000, frame_skip=4,
                 store_on_gpu=True,
                 device="cpu") -> None:
        self.store_on_gpu = store_on_gpu
        self.device = device
        self.action_dims = action_dims # Raw action dims
        self.frame_skip = frame_skip
        
        #shape
        self._obs_shape = (max_length//num_envs, num_envs, *obs_shape)
        self._action_shape = (max_length//num_envs, num_envs, len(action_dims))
        self._reward_shape = (max_length//num_envs, num_envs)
        self._termination_shape = (max_length//num_envs, num_envs)
        
        if self.store_on_gpu:
            self.obs_buffer = torch.empty(self._obs_shape, dtype=torch.uint8, device=device, requires_grad=False)
            self.action_buffer = torch.empty(self._action_shape, dtype=torch.uint8, device=device, requires_grad=False)
            self.reward_buffer = torch.empty(self._reward_shape, dtype=torch.float32, device=device, requires_grad=False)
            self.termination_buffer = torch.empty(self._termination_shape, dtype=torch.float32, device=device, requires_grad=False)
        else:
            self.obs_buffer = np.empty(self._obs_shape, dtype=np.uint8)
            self.action_buffer = np.empty(self._action_shape, dtype=np.float32)
            self.reward_buffer = np.empty(self._reward_shape, dtype=np.float32)
            self.termination_buffer = np.empty(self._termination_shape, dtype=np.float32)

        self.length = 0
        self.num_envs = num_envs
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None
        
    def __str__(self):
        def info(buf):
            if isinstance(buf, torch.Tensor):
                return f"Tensor(shape={tuple(buf.shape)}, dtype={buf.dtype}, device={buf.device})"
            elif isinstance(buf, np.ndarray):
                return f"ndarray(shape={buf.shape}, dtype={buf.dtype})"
            else:
                return str(type(buf))

        return (
            f"ReplayBuffer Status:\n"
            f"  Device: {'GPU' if self.store_on_gpu else 'CPU'}\n"
            f"  Observation Buffer: {info(self.obs_buffer)}\n"
            f"  Action Buffer:      {info(self.action_buffer)}\n"
            f"  Reward Buffer:      {info(self.reward_buffer)}\n"
            f"  Termination Buffer: {info(self.termination_buffer)}\n"
            f"  Total Steps Stored: {self.length * self.num_envs}\n"
            f"  Max Capacity:       {self.max_length}\n"
            f"  Warmup Steps:       {self.warmup_length}"
            f"  Ready:              {self.ready()}"
        )
    
    def ready(self):
        return self.length * self.num_envs > self.warmup_length
    
    def load_trajectory(self, path):
        buffer = pickle.load(open(path, "rb"))
        if self.store_on_gpu:
            self.external_buffer = {name: torch.from_numpy(buffer[name]).to(self.device) for name in buffer}
        else:
            self.external_buffer = buffer
        self.external_buffer_length = self.external_buffer["obs"].shape[0]

    def sample_external(self, batch_size, batch_length):
        indexes = np.random.randint(0, self.external_buffer_length+1-batch_length, size=batch_size)
        if self.store_on_gpu:
            obs = torch.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
            action = torch.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
            reward = torch.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
            termination = torch.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
        else:
            obs = np.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
            action = np.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
            reward = np.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
            termination = np.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
        return obs, action, reward, termination

    @torch.no_grad()
    def sample(self, batch_size, external_batch_size, batch_length, to_device="cuda"):
        assert (batch_length % 4) == 0
        if self.store_on_gpu:
            obs, action, reward, termination = [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    valid_range = (self.length + 1 - batch_length) // self.frame_skip
                    indexes = np.random.randint(0, valid_range, size=batch_size // self.num_envs) * self.frame_skip
                    # indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs) 
                    obs.append(torch.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    action.append(torch.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    reward.append(torch.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    termination.append(torch.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))

            if self.external_buffer_length is not None and external_batch_size > 0:
                external_obs, external_action, external_reward, external_termination = self.sample_external(
                    external_batch_size, batch_length, to_device)
                obs.append(external_obs)
                action.append(external_action)
                reward.append(external_reward)
                termination.append(external_termination)

            obs = torch.cat(obs, dim=0).float()
            obs = rearrange(obs, "B T H W C -> B C T H W")
            action = torch.cat(action, dim=0)
            reward = torch.cat(reward, dim=0)
            termination = torch.cat(termination, dim=0)
        else:
            obs, action, reward, termination = [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    valid_range = (self.length + 1 - batch_length) // self.frame_skip
                    indexes = np.random.randint(0, valid_range, size=batch_size // self.num_envs) * self.frame_skip
                    # indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs) 
                    obs.append(np.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    action.append(np.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    reward.append(np.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    termination.append(np.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))

            if self.external_buffer_length is not None and external_batch_size > 0:
                external_obs, external_action, external_reward, external_termination = self.sample_external(
                    external_batch_size, batch_length, to_device)
                obs.append(external_obs)
                action.append(external_action)
                reward.append(external_reward)
                termination.append(external_termination)

            obs = rearrange(torch.from_numpy(np.concatenate(obs, axis=0)).float().to(to_device) / 255, "B T H W C -> B C T H W")
            action = torch.from_numpy(np.concatenate(action, axis=0)).to(to_device)
            reward = torch.from_numpy(np.concatenate(reward, axis=0)).to(to_device)
            termination = torch.from_numpy(np.concatenate(termination, axis=0)).to(to_device)
        return obs, action, reward, termination

    def append(self, obs, action, reward, termination):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool

        save_obs = np.transpose(np.expand_dims(obs, axis=0), (0, 2, 3, 1)).copy() # [C H W] -> [1 C H W] -> [1 H W C]
        save_action = action # shape(8,)
        save_reward = np.array([reward])
        save_termination = np.array([termination])
        self.last_pointer = (self.last_pointer + 1) % (self.max_length//self.num_envs)
        if self.store_on_gpu:
            self.obs_buffer[self.last_pointer] = torch.from_numpy(save_obs)
            self.action_buffer[self.last_pointer] = torch.from_numpy(save_action)
            self.reward_buffer[self.last_pointer] = torch.from_numpy(save_reward)
            self.termination_buffer[self.last_pointer] = torch.from_numpy(save_termination)
        else:
            self.obs_buffer[self.last_pointer] = save_obs
            self.action_buffer[self.last_pointer] = save_action
            self.reward_buffer[self.last_pointer] = save_reward
            self.termination_buffer[self.last_pointer] = save_termination

        if len(self) < self.max_length:
            self.length += 1

    def export_buffer(self, file_path):
        if self.store_on_gpu:
            obs = self.obs_buffer[:self.length].cpu().numpy()
            action = self.action_buffer[:self.length].cpu().numpy()
            reward = self.reward_buffer[:self.length].cpu().numpy()
            done = self.termination_buffer[:self.length].cpu().numpy()
        else:
            obs = self.obs_buffer[:self.length]
            action = self.action_buffer[:self.length]
            reward = self.reward_buffer[:self.length]
            done = self.termination_buffer[:self.length]

        np.savez_compressed(file_path,
                            obs=obs,
                            action=action,
                            reward=reward,
                            done=done)
        print(f"Buffer exported to {file_path} (npz compressed)")

    def load_buffer(self, path):
        buffer = np.load(path)

        self.length = buffer["obs"].shape[0]//2
        self.external_buffer_length = None  # reset
        self.last_pointer = self.length - 1

        obs = buffer["obs"][:self.length]
        action = buffer["action"][:self.length]
        reward = buffer["reward"][:self.length]
        done = buffer["done"][:self.length]

        if self.store_on_gpu:
            self.obs_buffer[:self.length] = torch.from_numpy(obs).to(self.device)
            self.action_buffer[:self.length] = torch.from_numpy(action).to(self.device)
            self.reward_buffer[:self.length] = torch.from_numpy(reward).to(self.device)
            self.termination_buffer[:self.length] = torch.from_numpy(done).to(self.device)
        else:
            self.obs_buffer[:self.length] = obs
            self.action_buffer[:self.length] = action
            self.reward_buffer[:self.length] = reward
            self.termination_buffer[:self.length] = done
        print(f"Buffer loaded from {path}, length={self.length}")


    def __len__(self):
        return self.length * self.num_envs