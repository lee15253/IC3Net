import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler


class QBNTrainer():
    def __init__(self, policy_net, obs_qb_net, comm_qb_net, hidden_qb_net, storage, logdir):
        self.policy_net = policy_net
        self.obs_qb_net = obs_qb_net
        self.comm_qb_net = comm_qb_net
        self.hidden_qb_net = hidden_qb_net
        self.storage = storage
        self.writer = SummaryWriter(logdir)

    def train_all(self, batch_size, epochs, verbose=True):
        self.train_qb_net(self.obs_qb_net, batch_size, epochs, net_type='ob', verbose=verbose)
        self.train_qb_net(self.comm_qb_net, batch_size, epochs, net_type='comm', verbose=verbose)
        self.train_qb_net(self.hidden_qb_net, batch_size, epochs, net_type='hidden', verbose=verbose)

    def test_all(self):
        pass

    def train_qb_net(self, qb_net, batch_size, epochs, net_type='ob', verbose=True, val_ratio=0.1):
        mse_loss = nn.MSELoss().cuda()
        min_loss, best_perf = None, None
        optimizer = torch.optim.Adam(qb_net.parameters(), lr=1e-4, weight_decay=0)
        indices = np.arange(len(self.storage))
        np.random.shuffle(indices)
        split = int(np.floor(val_ratio * len(self.storage)))
        train_sampler = BatchSampler(SubsetRandomSampler(indices[split:]), batch_size, drop_last=False)
        val_sampler = BatchSampler(SubsetRandomSampler(indices[:split]), batch_size, drop_last=False)
        qb_net.to(self.storage.device)

        update_step = 0
        for epoch in range(epochs):
            qb_net.train()
            for train_batch_indices in train_sampler:
                input = self.storage.fetch_train_data(train_batch_indices, net_type)
                target = input.clone()
                pred, _ = qb_net(input)

                optimizer.zero_grad()
                loss = mse_loss(pred, target)
                loss.backward()
                optimizer.step()
                update_step += 1
                self.writer.add_scalar(net_type + '_loss', loss.item(), update_step)
            print(net_type, epoch)



