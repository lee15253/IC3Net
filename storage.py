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

    def fetch_train_data(self, indices, net_type='ob'):
        if net_type == 'ob':
            data = torch.FloatTensor(self.o_t_batch[indices]).to(self.device)
        elif net_type == 'comm':
            data = torch.FloatTensor(self.c_t_batch[indices]).to(self.device)
        elif net_type == 'hidden':
            data = torch.FloatTensor(self.h_t_batch[indices]).to(self.device)
        else:
            raise NotImplementedError
        return data
