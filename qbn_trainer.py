import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler
from action_utils import *
from utils import *


class QBNTrainer():
    def __init__(self, args, env, policy_net, obs_qb_net, comm_qb_net, hidden_qb_net, storage, writer):
        self.args = args
        self.env = env
        self.policy_net = policy_net
        self.obs_qb_net = obs_qb_net
        self.comm_qb_net = comm_qb_net
        self.hidden_qb_net = hidden_qb_net
        self.storage = storage
        self.writer = writer

    def train_all(self, verbose=True):
        self.train_qb_net(self.obs_qb_net,  net_type='ob', verbose=verbose)
        self.train_qb_net(self.comm_qb_net, net_type='comm', verbose=verbose)
        self.train_qb_net(self.hidden_qb_net, net_type='hidden', verbose=verbose)

    def test_all(self):
        # TODO: test performance when every network is repalced?
        pass

    def train_qb_net(self, qb_net, net_type='ob', verbose=True, val_ratio=0.1):
        mse_loss = nn.MSELoss().cuda()
        min_loss, best_perf = None, None
        optimizer = torch.optim.Adam(qb_net.parameters(), lr=1e-5, weight_decay=0)
        indices = np.arange(len(self.storage))
        np.random.shuffle(indices)
        split = int(np.floor(val_ratio * len(self.storage)))
        train_sampler = BatchSampler(SubsetRandomSampler(indices[split:]), self.args.batch_size, drop_last=False)
        val_sampler = BatchSampler(SubsetRandomSampler(indices[:split]), self.args.batch_size, drop_last=False)

        for epoch in range(self.args.epochs):
            qb_net.to(self.storage.device)
            # Validation
            train_losses, val_losses = [], []
            qb_net.train()
            for train_batch_indices in train_sampler:
                input = self.storage.fetch_train_data(train_batch_indices, net_type)
                target = input.clone()
                pred, _ = qb_net(input)

                optimizer.zero_grad()
                loss = mse_loss(pred, target)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            self.writer.add_scalar(net_type + '/train_loss', np.mean(train_losses), epoch)

            # Validation
            qb_net.eval()
            with torch.no_grad():
                for val_batch_indices in val_sampler:
                    input = self.storage.fetch_train_data(val_batch_indices, net_type)
                    target = input.clone()
                    pred, _ = qb_net(input)
                    loss = mse_loss(pred, target)
                    val_losses.append(loss.item())
                self.writer.add_scalar(net_type + '/val_loss', np.mean(val_losses), epoch)

            # TODO: Test with environment
            if net_type == 'ob':
                self.perform_rollouts(self.args.num_test_rollout_steps,
                                      net_type=net_type,
                                      obs_qb_net=qb_net.to('cpu'),
                                      epoch=epoch)
            elif net_type =='comm':
                self.perform_rollouts(self.args.num_test_rollout_steps,
                                      net_type=net_type,
                                      comm_qb_net=qb_net.to('cpu'),
                                      epoch=epoch)
            elif net_type =='hidden':
                self.perform_rollouts(self.args.num_test_rollout_steps,
                                      net_type=net_type,
                                      hidden_qb_net=qb_net.to('cpu'),
                                      epoch=epoch)

    def perform_rollouts(self, num_rollout_steps, net_type='policy', store=False, epoch=0,
                         obs_qb_net=None, comm_qb_net=None, hidden_qb_net=None):
        batch = []
        stats = dict()
        stats['num_episodes'] = 0
        while len(batch) < num_rollout_steps:
            if self.args.noisy_rollouts:
                # TODO: implement noisy rollouts
                raise NotImplementedError
            else:
                episode, episode_stat = self.get_episode(obs_qb_net, comm_qb_net, hidden_qb_net)
            merge_stat(episode_stat, stats)
            stats['num_episodes'] += 1
            batch += episode
        stats['num_steps'] = len(batch)
        if store:
            self.storage.store(rollouts=batch)
        self.writer.add_scalar(net_type + '/avg_rewards', np.mean(stats['reward']) / stats['num_episodes'], epoch)
        self.writer.add_scalar(net_type + '/success_rate', stats['success'] / stats['num_episodes'], epoch)

    def get_episode(self, obs_qb_net, comm_qb_net, hidden_qb_net):
        episode = []
        stat = dict()
        info = dict()
        state = self.env.reset(epoch=0)
        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        for t in range(self.args.max_steps):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            # recurrence over time
            if self.args.rnn_type == 'LSTM' and t == 0:
                prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

            x = [state, prev_hid]
            action_out, value, prev_hid, latent = self.policy_net(x, info, obs_qb_net, comm_qb_net, hidden_qb_net)

            if (t + 1) % self.args.detach_gap == 0:
                if self.args.rnn_type == 'LSTM':
                    prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                else:
                    prev_hid = prev_hid.detach()

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            next_state, reward, done, info = self.env.step(actual)

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents,
                                                                                               dtype=int)

                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm'] = stat.get('enemy_comm', 0) + info['comm_action'][self.args.nfriendly:]

            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1
            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)
            episode.append(latent)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return episode, stat


    def finetune(self):
        pass




