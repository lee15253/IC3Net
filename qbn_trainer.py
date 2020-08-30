import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler
from action_utils import *
from models import *
from utils import *
import ipdb

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask',
                                       'next_state','reward', 'misc', 'latent'))

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

    def finetune(self, val_ratio=0.1):
        # Load best-performing Quantized Model
        self.obs_qb_net.load_state_dict(torch.load(self.writer.log_dir + '/ob.pth'))
        self.comm_qb_net.load_state_dict(torch.load(self.writer.log_dir + '/comm.pth'))
        self.hidden_qb_net.load_state_dict(torch.load(self.writer.log_dir + '/hidden.pth'))

        best_perf = -float('inf')
        model_path = self.writer.log_dir + '/mmn.pth'
        mm_net = MMNet(self.policy_net, self.obs_qb_net, self.comm_qb_net, self.hidden_qb_net)
        # Loss function & Optimizer
        #mse_loss = nn.MSELoss().cuda()
        kl_div_loss = nn.KLDivLoss(size_average=False).cuda()
        optimizer = torch.optim.Adam(mm_net.parameters(), lr=1e-4, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.qbn_epochs+1,
                                                               eta_min=0, last_epoch=-1)
        # Split Train / Validation
        indices = np.arange(len(self.storage))
        np.random.shuffle(indices)
        split = int(np.floor(val_ratio * len(self.storage)))
        train_sampler = BatchSampler(SubsetRandomSampler(indices[split:]), self.args.batch_size, drop_last=False)
        val_sampler = BatchSampler(SubsetRandomSampler(indices[:split]), self.args.batch_size, drop_last=False)

        # Test initial performance
        mm_net.to('cpu')
        self.perform_rollouts(mm_net, self.args.num_test_rollout_steps, net_type='Quantized', epoch=0)
        torch.save(mm_net.state_dict(), model_path)

        for epoch in range(self.args.finetune_epochs):
            train_losses, val_losses = [], []
            # Train
            mm_net.train()
            for train_batch_indices in train_sampler:
                mm_net.to('cpu')
                optimizer.zero_grad()
                for train_index in train_batch_indices:
                    (x_t, cell_t, h_t, a_t, ac_t, info_t) \
                        = self.storage.fetch_train_data(train_index, 'mm')
                    x_t = [x_t.unsqueeze(0), (h_t, cell_t)]
                    action, _, _, _ = mm_net(x_t, info_t)
                    # a_t & target-action is the log-softmax logit
                    #loss = mse_loss(a_t, target_action[0].squeeze(0))
                    target_action_prob = torch.exp(a_t)
                    target_comm_prob = torch.exp(ac_t)
                    loss = kl_div_loss(action[0].squeeze(0), target_action_prob) \
                            + kl_div_loss(action[1].squeeze(0), target_comm_prob)
                    loss.backward()
                    train_losses.append(loss.item())

                mm_net.to('cuda')
                torch.nn.utils.clip_grad_norm_(mm_net.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
            self.writer.add_scalar('Quantized/train_loss', np.mean(train_losses), epoch)

            # Validation
            mm_net.eval()
            with torch.no_grad():
                for val_batch_indices in val_sampler:
                    mm_net.to('cpu')
                    for val_index in val_batch_indices:
                        (x_t, cell_t, h_t, a_t, ac_t, info_t) \
                            = self.storage.fetch_train_data(val_index, 'mm')
                        x_t = [x_t.unsqueeze(0), (h_t, cell_t)]
                        action, _, _, _ = mm_net(x_t, info_t)
                        #loss = mse_loss(a_t, target_action[0].squeeze(0))
                        target_action_prob = torch.exp(a_t)
                        target_comm_prob = torch.exp(ac_t)
                        loss = kl_div_loss(action[0].squeeze(0), target_action_prob) \
                                + kl_div_loss(action[1].squeeze(0), target_comm_prob)
                        val_losses.append(loss.item())
                self.writer.add_scalar('Quantized/val_loss', np.mean(val_losses), epoch)

            # Performance Test
            avg_rewards = self.perform_rollouts(mm_net, self.args.num_test_rollout_steps, net_type='Quantized', epoch=epoch+1)
            if avg_rewards > best_perf:
                torch.save(mm_net.state_dict(), model_path)
                best_perf = avg_rewards


    def train_qb_net(self, qb_net, net_type='ob', verbose=True, val_ratio=0.1):
        best_perf = -float('inf')
        model_path = self.writer.log_dir + '/' + net_type + '.pth'
        # Loss function & Optimizer
        mse_loss = nn.MSELoss().cuda()
        optimizer = torch.optim.Adam(qb_net.parameters(), lr=1e-4, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.qbn_epochs+1,
                                                               eta_min=0, last_epoch=-1)
        # Split Train / Validation
        indices = np.arange(len(self.storage))
        np.random.shuffle(indices)
        split = int(np.floor(val_ratio * len(self.storage)))
        batch_size = self.args.batch_size // self.storage.n_agents
        train_sampler = BatchSampler(SubsetRandomSampler(indices[split:]), batch_size, drop_last=False)
        val_sampler = BatchSampler(SubsetRandomSampler(indices[:split]), batch_size, drop_last=False)

        # Test initial performance
        qb_net.eval()
        if net_type == 'ob':
            mm_net = MMNet(self.policy_net, obs_qb_net=qb_net.to('cpu'))
        elif net_type == 'comm':
            mm_net = MMNet(self.policy_net, comm_qb_net=qb_net.to('cpu'))
        elif net_type == 'hidden':
            mm_net = MMNet(self.policy_net, hidden_qb_net=qb_net.to('cpu'))
        else:
            raise NotImplementedError
        self.perform_rollouts(mm_net, self.args.num_test_rollout_steps, net_type=net_type, epoch=0)

        for epoch in range(self.args.qbn_epochs):
            train_losses, val_losses = [], []
            # Train
            qb_net.to(self.storage.device)
            qb_net.train()
            for train_batch_indices in train_sampler:
                input = self.storage.fetch_train_data(train_batch_indices, net_type)
                target = input.clone()
                pred, _ = qb_net(input)

                optimizer.zero_grad()
                loss = mse_loss(pred, target)
                loss.backward()
                optimizer.step()
                scheduler.step()
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

            # Test performance with rollouts
            if net_type == 'ob':
                mm_net = MMNet(self.policy_net, obs_qb_net=qb_net.to('cpu'))
            elif net_type == 'comm':
                mm_net = MMNet(self.policy_net, comm_qb_net=qb_net.to('cpu'))
            elif net_type == 'hidden':
                mm_net = MMNet(self.policy_net, hidden_qb_net=qb_net.to('cpu'))
            else:
                raise NotImplementedError
            avg_rewards = self.perform_rollouts(mm_net, self.args.num_test_rollout_steps,
                                                net_type=net_type, epoch=epoch+1)
            if avg_rewards > best_perf:
                torch.save(qb_net.state_dict(), model_path)
                best_perf = avg_rewards

    def perform_rollouts(self, mm_net, num_rollout_steps, net_type='policy', store=False, epoch=0):
        batch = []
        stats = dict()
        stats['num_episodes'] = 0
        print('total_rollout:',num_rollout_steps)
        while len(batch) < num_rollout_steps:
            print("len(rollout):",len(batch))
            if self.args.noisy_rollouts:
                # TODO: implement noisy rollouts
                raise NotImplementedError
            else:
                episode, episode_stat = self.get_episode(mm_net)
            merge_stat(episode_stat, stats)
            stats['num_episodes'] += 1
            batch += episode
        stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))

        if store:
            latent = batch[-1]
            self.storage.store(rollouts=latent)
        avg_rewards = np.mean(stats['reward']) / stats['num_episodes']
        self.writer.add_scalar(net_type + '/avg_rewards', avg_rewards, epoch)
        if 'success' in stats:
            self.writer.add_scalar(net_type + '/success_rate', stats['success'] / stats['num_episodes'], epoch)
        return avg_rewards

    def get_episode(self, mm_net):
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
                prev_hid = mm_net.policy_net.init_hidden(batch_size=state.shape[0])

            x = [state, prev_hid]
            action_out, value, prev_hid, latent = mm_net(x, info)
            if (t + 1) % self.args.detach_gap == 0:
                if self.args.rnn_type == 'LSTM':
                    prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                else:
                    prev_hid = prev_hid.detach()
            
            action = select_action(self.args, action_out)
            # BK
            latent['actual_a_t'] = action[0].squeeze(0)
            latent['curr_t'] = t
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
            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state,
                               reward, misc, latent)
            episode.append(trans)
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
