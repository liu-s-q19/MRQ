import numpy as np
import torch

class PrioritizedReplayBuffer:
    def __init__(self, obs_shape, action_dim, max_action, pixel_obs, device, history=1, horizon=1, max_size=1e6, batch_size=256, alpha=0.6, beta=0.4, beta_increment=0.001, initial_priority=1.0):
        self.max_size = int(max_size)
        self.batch_size = batch_size
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        self.history = history
        self.horizon = horizon
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = initial_priority
        self.position = 0
        self.size = 0
        self.env_terminates = False
        self.state_shape = [obs_shape[0] * history]
        if pixel_obs:
            self.state_shape += [obs_shape[1], obs_shape[2]]
        self.obs_dtype = np.uint8 if pixel_obs else np.float32
        # 自动判断是否用GPU存储
        try:
            import torch
            mem, _ = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
            obs_space = np.prod((self.max_size, *self.state_shape))
            if pixel_obs:
                obs_space = obs_space * 1
            else:
                obs_space = obs_space * 4
            ard_space = self.max_size * (action_dim + 2) * 4
            if obs_space + ard_space < mem:
                self.storage_device = self.device
            else:
                self.storage_device = torch.device('cpu')
        except:
            self.storage_device = torch.device('cpu')
        self.obs = np.zeros((self.max_size, *self.state_shape), dtype=self.obs_dtype)
        self.actions = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.max_size, *self.state_shape), dtype=self.obs_dtype)
        self.rewards = np.zeros(self.max_size, dtype=np.float32)
        self.terminateds = np.zeros(self.max_size, dtype=np.bool_)
        self.truncateds = np.zeros(self.max_size, dtype=np.bool_)
        self.priorities = np.full(self.max_size, initial_priority, dtype=np.float32)
        self.history_queue = [0 for _ in range(self.history)]
        self.state_ind = np.zeros((self.max_size, self.history), dtype=np.int32)
        self.next_ind = np.zeros((self.max_size, self.history), dtype=np.int32)

    def add(self, state, action, next_state, reward, terminated, truncated):
        self.obs[self.position] = state
        self.actions[self.position] = action
        self.next_obs[self.position] = next_state
        self.rewards[self.position] = reward
        self.terminateds[self.position] = bool(terminated)
        self.truncateds[self.position] = bool(truncated)
        self.priorities[self.position] = self.max_priority
        self.state_ind[self.position] = np.array(self.history_queue, dtype=np.int32)
        next_pos = (self.position + 1) % self.max_size
        self.history_queue.append(next_pos)
        self.history_queue = self.history_queue[-self.history:]
        self.next_ind[self.position] = np.array(self.history_queue, dtype=np.int32)
        self.position = next_pos
        self.size = min(self.size + 1, self.max_size)
        if terminated:
            self.env_terminates = True

    def sample(self, horizon, include_intermediate=False):
        valid_size = self.size - horizon
        probs = self.priorities[:valid_size] ** self.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(valid_size, self.batch_size, p=probs, replace=False)
        ind = (indices.reshape(-1,1) + np.arange(horizon).reshape(1,-1)) % self.max_size
        state_ind = self.state_ind[ind]
        next_state_ind = self.next_ind[ind[:,-1].reshape(-1,1)]
        if include_intermediate:
            cat_idx = np.concatenate([state_ind, next_state_ind], 1)
            both_state = np.take(self.obs, cat_idx, axis=0).reshape(self.batch_size,-1,*self.state_shape)
            state = torch.as_tensor(both_state[:,:-1], dtype=torch.float, device=self.device)
            next_state = torch.as_tensor(both_state[:,1:], dtype=torch.float, device=self.device)
            action = torch.as_tensor(np.take(self.actions, ind, axis=0), dtype=torch.float, device=self.device)
            reward = torch.as_tensor(np.take(self.rewards, ind, axis=0), dtype=torch.float, device=self.device)
            not_done = torch.as_tensor(~(np.take(self.terminateds, ind, axis=0) | np.take(self.truncateds, ind, axis=0)), dtype=torch.float, device=self.device)
        else:
            state_indices = self.state_ind[ind[:,0], 0]
            next_indices = self.next_ind[ind[:,-1], 0]
            state = torch.as_tensor(np.take(self.obs, state_indices, axis=0), dtype=torch.float, device=self.device)
            next_state = torch.as_tensor(np.take(self.obs, next_indices, axis=0), dtype=torch.float, device=self.device)
            action = torch.as_tensor(np.take(self.actions, ind[:,0], axis=0), dtype=torch.float, device=self.device)
            reward = torch.as_tensor(np.take(self.rewards, ind, axis=0), dtype=torch.float, device=self.device)
            not_done = torch.as_tensor(~(np.take(self.terminateds, ind, axis=0) | np.take(self.truncateds, ind, axis=0)), dtype=torch.float, device=self.device)
        reward = reward.unsqueeze(-1) if reward.ndim == 2 else reward
        not_done = not_done.unsqueeze(-1) if not_done.ndim == 2 else not_done
        self.last_sampled_indices = indices
        return state, action, next_state, reward, not_done

    def update_priority(self, priority):
        for idx, p in zip(self.last_sampled_indices, priority):
            self.priorities[idx] = float(p)
            self.max_priority = max(self.max_priority, float(p))

    def reward_scale(self, eps=1e-8):
        return float(np.abs(self.rewards[:self.size]).mean().clip(min=eps))

    def save(self, save_folder):
        np.savez_compressed(f'{save_folder}/prio_buffer_data',
            obs=self.obs,
            actions=self.actions,
            next_obs=self.next_obs,
            rewards=self.rewards,
            terminateds=self.terminateds,
            truncateds=self.truncateds,
            priorities=self.priorities,
            position=self.position,
            size=self.size,
            state_ind=self.state_ind,
            next_ind=self.next_ind,
            history_queue=np.array(self.history_queue),
            env_terminates=self.env_terminates
        )

    def load(self, save_folder):
        data = np.load(f'{save_folder}/prio_buffer_data.npz', allow_pickle=True)
        self.obs = data['obs']
        self.actions = data['actions']
        self.next_obs = data['next_obs']
        self.rewards = data['rewards']
        self.terminateds = data['terminateds']
        self.truncateds = data['truncateds']
        self.priorities = data['priorities']
        self.position = int(data['position'])
        self.size = int(data['size'])
        self.state_ind = data['state_ind']
        self.next_ind = data['next_ind']
        self.history_queue = list(data['history_queue'])
        self.env_terminates = bool(data['env_terminates']) 