import torch
from collections import deque
import ipdb

class Storage():
    def __init__(self, args, observation_dim):
        self.storage_size = args.storage_size
        self.n_agents = args.nagents
        self.observation_dim = observation_dim
        self.hid_size = args.hid_size
        self.num_actions = args.num_actions[0]
        self.device = torch.device('cuda')
        self.idx = 0
        self.FSM = args.generate_FSM
        
        self.x_t_batch = torch.zeros(self.storage_size, self.n_agents, self.observation_dim)
        self.info_t_batch = deque([], maxlen=self.storage_size)
        self.cell_t_batch = torch.zeros(self.storage_size, self.n_agents, self.hid_size)
        self.h_t_batch = torch.zeros(self.storage_size, self.n_agents, self.hid_size)
        self.o_t_batch = torch.zeros(self.storage_size, self.n_agents, self.hid_size)
        self.c_t_batch = torch.zeros(self.storage_size, self.n_agents, self.hid_size)
        self.a_t_batch = torch.zeros(self.storage_size, self.n_agents, self.num_actions)
        self.ac_t_batch = torch.zeros(self.storage_size, self.n_agents, 2)
        self.h_t1_batch = torch.zeros(self.storage_size, self.n_agents, self.hid_size)

        # BK: Used in generating FSM
        if self.FSM:
            self.q_x_size = args.obs_quantize_size
            self.q_c_size = args.comm_quantize_size
            self.q_h_size = args.hidden_quantize_size
            self.q_x_batch = torch.zeros(self.storage_size, self.n_agents, self.q_x_size)
            self.q_c_batch = torch.zeros(self.storage_size, self.n_agents, self.q_c_size)
            self.q_h_batch = torch.zeros(self.storage_size, self.n_agents, self.q_h_size)
            self.actual_a_t_batch = torch.zeros(self.storage_size, self.n_agents)

    def store(self, rollouts):
        for step in range(len(rollouts)):
            for agent_idx in range(self.n_agents):
                self.x_t_batch[self.idx][agent_idx] = rollouts[step]['x_t'][agent_idx]
                self.cell_t_batch[self.idx][agent_idx] = rollouts[step]['cell_t'][agent_idx]
                self.h_t_batch[self.idx][agent_idx] = rollouts[step]['h_t'][agent_idx]
                self.o_t_batch[self.idx][agent_idx] = rollouts[step]['o_t'][agent_idx]
                self.c_t_batch[self.idx][agent_idx] = rollouts[step]['c_t'][agent_idx]
                self.a_t_batch[self.idx][agent_idx] = rollouts[step]['a_t'][agent_idx]
                self.ac_t_batch[self.idx][agent_idx] = rollouts[step]['ac_t'][agent_idx]
                self.h_t1_batch[self.idx][agent_idx] = rollouts[step]['h_t1'][agent_idx]

                # BK: Used in generating FSM
                if self.FSM:
                    self.q_x_batch[self.idx][agent_idx] = rollouts[step]['q_x'][agent_idx]
                    self.q_c_batch[self.idx][agent_idx] = rollouts[step]['q_c'][agent_idx]
                    self.q_h_batch[self.idx][agent_idx] = rollouts[step]['q_h'][agent_idx]
                    # self.actual_a_t_batch[self.idx][agent_idx] = rollouts[step]['actual_a_t'][agent_idx]

            self.info_t_batch.append(rollouts[step]['info_t'])
            self.idx = (self.idx + 1) % self.storage_size

    def __len__(self):
        return self.idx

    def fetch_train_data(self, indices, net_type='ob'):
        if net_type == 'ob':
            data = self.o_t_batch[indices].reshape(-1, self.hid_size).to(self.device).detach()
        elif net_type == 'comm':
            data = self.c_t_batch[indices].reshape(-1, self.hid_size).to(self.device).detach()
        elif net_type == 'hidden':
            data = self.h_t_batch[indices].reshape(-1, self.hid_size).to(self.device).detach()
        elif net_type == 'mm':
            x_t_batch = self.x_t_batch[indices].to('cpu').detach()
            cell_t_batch = self.cell_t_batch[indices].to('cpu').detach()
            h_t_batch = self.h_t_batch[indices].to('cpu').detach()
            a_t_batch = self.a_t_batch[indices].to('cpu').detach()
            ac_t_batch = self.ac_t_batch[indices].to('cpu').detach()
            info_t_batch = self.info_t_batch[indices]
            data = (x_t_batch, cell_t_batch, h_t_batch, a_t_batch, ac_t_batch, info_t_batch)
        else:
            raise NotImplementedError
        return data

    # BK: Used in generating FSM
    def fetch_fsm_data(self):
        # agent 1's quantized_(obs,comm,hidden)
        q_x_batch = self.q_x_batch[:self.idx,0,:].to('cpu').detach().numpy()
        q_c_batch = self.q_c_batch[:self.idx,0,:].to('cpu').detach().numpy()
        q_h_batch = self.q_h_batch[:self.idx,0,:].to('cpu').detach().numpy()
        a_t_batch = self.a_t_batch[:self.idx,0,:].to('cpu').detach().numpy()
        # a_t_batch = self.actual_a_t_batch[:self.idx,0].to('cpu').detach().numpy()
        return (q_x_batch, q_c_batch, q_h_batch, a_t_batch)  