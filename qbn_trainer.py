import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class QBNTrainer():
    def __init__(self, policy_net, obs_qb_net, comm_qb_net, hidden_qb_net, storage):
        self.policy_net = policy_net
        self.obs_qb_net = obs_qb_net
        self.comm_qb_net = comm_qb_net
        self.hidden_qb_net = hidden_qb_net
        self.storage = storage

    def train_obs_qb_net(self, batch_size, epochs, verbose=True):
        self.train_qb_net(self.obs_qb_net, batch_size, epochs, verbose)
        self.test_qb_net(obs_qb_net=self.obs_qb_net)

    def train_comm_qb_net(self, batch_size, epochs, verbose=True):
        self.train_qb_net(self.comm_qb_net, batch_size, epochs, verbose)
        self.test_qb_net(comm_qb_net=self.comm_qb_net)

    def train_hidden_qb_net(self, batch_size, epochs, verbose=True):
        self.train_qb_net(self.hidden_qb_net, batch_size, epochs, verbose)
        self.test_qb_net(hidden_qb_net=self.hidden_qb_net)

    def test_all(self):
        self.test_qb_net(self.obs_qb_net, self.comm_qb_net, self.hidden_qb_net)

    def train_qb_net(self, qb_net, batch_size, epochs, verbose=True):
        mse_loss = nn.MSELoss().cuda()
        min_loss, best_perf = None, None
        optimizer = torch.optim.Adam(qb_net.paramters(), lr=1e-4, weight_decay=0)

    def test_qb_net(self, obs_qb_net=None, comm_qb_net=None, hidden_qb_net=None):
        pass
