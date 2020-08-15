import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Storage():
    def __init__(self, storage_size, n_agents, hid_size, num_actions):
        self.storage_size = storage_size
        self.n_agents = n_agents
        self.hid_size = hid_size
        self.num_actions = num_actions
        self.device = torch.device('cuda')
        self.idx = 0

        self.h_t_batch = torch.zeros(self.storage_size, self.hid_size)
        self.o_t_batch = torch.zeros(self.storage_size, self.hid_size)
        self.c_t_batch = torch.zeros(self.storage_size, self.hid_size)
        self.a_t_batch = torch.zeros(self.storage_size, self.num_actions)
        self.h_t1_batch = torch.zeros(self.storage_size, self.hid_size)

    def store(self, rollouts):
        for step in range(len(rollouts)):
            for agent_idx in range(self.n_agents):
                self.h_t_batch[self.idx] = rollouts[step]['h_t'][agent_idx]
                self.o_t_batch[self.idx] = rollouts[step]['o_t'][agent_idx]
                self.c_t_batch[self.idx] = rollouts[step]['c_t'][agent_idx]
                self.a_t_batch[self.idx] = rollouts[step]['a_t'][agent_idx]
                self.h_t1_batch[self.idx] = rollouts[step]['h_t1'][agent_idx]
                self.idx += 1

    def __len__(self):
        return self.idx

    def fetch_obs_data(self, indices):
        o_t_batch = torch.FloatTensor(self.o_t_batch[indices]).to(self.device)
        return o_t_batch

    def fetch_comm_data(self, indices):
        c_t_batch = torch.FloatTensor(self.c_t_batch[indices]).to(self.device)
        return c_t_batch

    def fetch_hidden_data(self, indices):
        h_t_batch = torch.FloatTensor(self.h_t_batch[indices]).to(self.device)
        return h_t_batch
