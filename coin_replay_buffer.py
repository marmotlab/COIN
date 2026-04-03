import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, obs_dim, state_dim, action_dim, agent_dim, max_size=int(2e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, obs_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_obs = np.zeros((max_size, obs_dim))
        self.reward = np.zeros((max_size, 1))
        self.global_reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.global_state = np.zeros((max_size, agent_dim, state_dim))
        self.global_action = np.zeros((max_size, agent_dim, action_dim))
        self.global_mask = np.zeros((max_size, agent_dim))
        self.next_global_state = np.zeros((max_size, agent_dim, state_dim))
        self.next_global_action = np.zeros((max_size, agent_dim, action_dim))
        self.next_global_mask = np.zeros((max_size, agent_dim))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, obs, action, next_obs, individual_r, global_r, done, global_s, global_a, global_mask, next_global_s, next_global_a, next_global_mask):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.reward[self.ptr] = individual_r
        self.global_reward[self.ptr] = global_r
        self.not_done[self.ptr] = 1. - done
        self.global_state[self.ptr] = global_s
        self.global_action[self.ptr] = global_a
        self.global_mask[self.ptr] = global_mask
        self.next_global_state[self.ptr] = next_global_s
        self.next_global_action[self.ptr] = next_global_a
        self.next_global_mask[self.ptr] = next_global_mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.obs[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_obs[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.global_reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.global_state[ind]).to(self.device),
            torch.FloatTensor(self.global_action[ind]).to(self.device),
            torch.FloatTensor(self.global_mask[ind]).to(self.device),
            torch.FloatTensor(self.next_global_state[ind]).to(self.device),
            torch.FloatTensor(self.next_global_action[ind]).to(self.device),
            torch.FloatTensor(self.next_global_mask[ind]).to(self.device)
        )
