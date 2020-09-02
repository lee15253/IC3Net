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
import os
import pickle
import sklearn.metrics.pairwise as pw
import itertools

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

    def perform_rollouts(self, mm_net, num_rollout_steps, net_type='policy', store=False, epoch=0, eval_mode=False):

        batch = []
        stats = dict()
        stats['num_episodes'] = 0
        print('total_rollout:',num_rollout_steps)
        
        while len(batch) < num_rollout_steps:
            print("\nlen(rollout):",len(batch))
            if self.args.noisy_rollouts:
                # TODO: implement noisy rollouts
                raise NotImplementedError
            else:
                episode, episode_stat = self.get_episode(mm_net, eval_mode, stats['num_episodes'])
            merge_stat(episode_stat, stats)
            stats['num_episodes'] += 1
            batch += episode

        stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        if store:
            latent = batch[-1]
            self.storage.store(rollouts=latent)
        avg_rewards = np.mean(stats['reward']) / stats['num_episodes']
        
        # BK: self.writer.add_scalar('eval_mode/avg_rewards', avg_rewards, epoch)
        self.writer.add_scalar(net_type + '/avg_rewards', avg_rewards, epoch)
        if 'success' in stats:
            # BK: self.writer.add_scalar('eval_mode/success_rate', stats['success'] / stats['num_episodes'], epoch)
            self.writer.add_scalar(net_type + '/success_rate', stats['success'] / stats['num_episodes'], epoch)
        return avg_rewards

    def get_episode(self, mm_net, eval_mode = False, num_episodes = 0):
        episode = []
        stat = dict()
        info = dict()
        state = self.env.reset(epoch=0)
        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        # BK
        if eval_mode:
            with open(os.path.join(self.args.load+self.args.dest,'minimized_files.p'), 'rb') as handle:
                m_files = pickle.load(handle)
            self.transaction = m_files[0]
            self.state_desc = m_files[1]
            self.state_space = m_files[2]
            self.start_state = m_files[3]
            self.obs_minobs_map = m_files[4]  # {'1_1':'1_1',  '10_13':'10_13',  '21_35':'1_1'(1_1로merge됨)}
            self.minobs_obs_map = m_files[5]  # {'1_1':['1_1', '21_35'], '10_13':[10_13,...]}
            self.minimized = m_files[6]
            self.obs_space = m_files[7]
            self.comm_space = m_files[8]
            self.original_state_space = m_files[9]

            random.seed(int(time.time()))
            np.random.seed(int(time.time()))
            torch.manual_seed(int(time.time()))
            mm_net.eval()


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

            # BK
            if eval_mode:
                if t==0:
                    curr_state = self.start_state

                # 1. 0번 agent의 obs, comm, hidden_t 각각 quantize하고 index 가져온다.
                curr_obs_x = latent['q_x'][0].detach().numpy()
                _, obs_index = self._get_index(self.obs_space, curr_obs_x, force=False)
                curr_comm_x = latent['q_c'][0].detach().numpy()
                _, comm_index = self._get_index(self.comm_space, curr_comm_x, force=False)

                # 둘다 각각 본 적이 있고, 그 조합도 본 적이 있다.
                if (obs_index != None) and (comm_index != None) and (str(obs_index)+'_'+str(comm_index) in self.obs_minobs_map):
                    qx_qc = self.obs_minobs_map[str(obs_index)+'_'+str(comm_index)]

                # 처음보는 obs_x의 경우, 가장 비슷한걸로 matching한다.
                elif obs_index == None and comm_index != None:
                    # obs_cos_sim = pw.cosine_similarity(np.expand_dims(curr_obs_x,0), self.obs_space)
                    # 1-1. minimize 이전에 해당 comm_index와 같이 나온적이 있는 모든 comm 후보를 불러온다.
                    every_obs_with_current_comm = [i for i in self.obs_minobs_map.keys() if i.split('_')[1] == str(comm_index)]
                    assert len(every_obs_with_current_comm) != 0, '해당 comm_index가 있는 조합이 없다고? error'
                    # 1-2. 그 중 현재 obs_x와 가장 cos_sim이 높은 obs_x를 선택한다.
                    possible_obs = [int(i.split('_')[0]) for i in every_obs_with_current_comm]
                    np_possible_obs = self.obs_space[np.array(possible_obs)]
                    obs_cos_sim = pw.cosine_similarity(np.expand_dims(curr_obs_x,0), np_possible_obs)
                    obs_index = possible_obs[np.argmax(obs_cos_sim)]
                    qx_qc = self.obs_minobs_map[str(obs_index)+'_'+str(comm_index)]
                    print('t:{}, 1.obs_index is replaced with {}(minimized:{})'.format(t,str(obs_index)+'_'+str(comm_index), qx_qc))

                # # 처음보는 comm_x의 경우, 가장 비슷한걸로 matching한다.
                elif obs_index != None and comm_index == None:  
                    # comm_cos_sim = pw.cosine_similarity(np.expand_dims(curr_comm_x,0), self.comm_space)
                    # 1-1. minimize 이전에 해당 obs_index와 같이 나온적이 있는 모든 comm 후보를 불러온다.
                    every_comm_with_current_obs = [i for i in self.obs_minobs_map.keys() if i.split('_')[0] == str(obs_index)]
                    assert len(every_comm_with_current_obs) != 0, '해당 obs_index가 있는 조합이 없다고? error'
                    # 1-2. 그 중 현재 comm_x와 가장 cos_sim이 높은 comm_x을 선택한다.
                    possible_comm = [int(i.split('_')[1]) for i in every_comm_with_current_obs]
                    np_possible_comm = self.comm_space[np.array(possible_comm)]
                    comm_cos_sim = pw.cosine_similarity(np.expand_dims(curr_comm_x,0) , np_possible_comm)
                    comm_index = possible_comm[np.argmax(comm_cos_sim)]
                    qx_qc = self.obs_minobs_map[str(obs_index)+'_'+str(comm_index)]
                    print('t:{}, 1.comm_index is replaced with {}(minimized:{})'.format(t,str(obs_index)+'_'+str(comm_index), qx_qc))

                else:
                    qx_qc = None
                    print('t:{}, 1.obs, comm index both None'.format(t))
                                            
                # 2. 구해진 qx_qc가 self.transaction[curr_state]에 없는 경우 가장 similarity가 높은 qx_qc로 바꾼다.
                if qx_qc not in self.transaction[curr_state]:         
                    candidate = [i for i in self.transaction[curr_state].keys() 
                                        if self.transaction[curr_state][i] != None]
                    candidate_all = [self.minobs_obs_map[i] for i in candidate]
                    candidate_all = list(set(itertools.chain.from_iterable(candidate_all)))
                    candidate_obs = [int(i.split('_')[0]) for i in candidate_all]
                    candidate_comm = [int(i.split('_')[1]) for i in candidate_all]
                    
                    obs_cos_sim = pw.cosine_similarity(np.expand_dims(curr_obs_x,0), self.obs_space[candidate_obs])
                    comm_cos_sim = pw.cosine_similarity(np.expand_dims(curr_comm_x,0), self.comm_space[candidate_comm])
                    sum_cos_sim = obs_cos_sim + comm_cos_sim
                    new_qx = candidate_obs[np.argmax(obs_cos_sim+comm_cos_sim)]
                    new_qc = candidate_comm[np.argmax(obs_cos_sim+comm_cos_sim)]
                    new_qx_qc = str(new_qx) + '_' + str(new_qc)
                    min_new_qx_qc = self.obs_minobs_map[new_qx_qc]
                    print('t:{}, 2.qx_qc is replaced from {} to {}(minimized:{})'.format(t, qx_qc, new_qx_qc, min_new_qx_qc))
                    qx_qc = min_new_qx_qc
                        
                # 3. transaction table을 이용하여 hidden_t+1을 계산한다.
                q_next_state = self.transaction[curr_state][qx_qc]
                assert q_next_state != None, 'None state encountered! Exit'

                # 4. state_desc를 이용해서 action을 계산한다. 1->1(2연break) 막아버린다
                if curr_state == 1 and qx_qc == '205_310':
                    print('1에서 1로가는 edge로 들어옴!')
                    ipdb.set_trace()
                    curr_state = 0
                else:
                    curr_state = q_next_state

                q_action = int(self.state_desc[curr_state]['action'])

                # 5. env.step(action)으로 next_obs를 받는다.
                # TODO: evaluation 시 policy가 softmax (not argmax) -> logit은 (0.86, 0.14)인데 후자가 선택되버림
                # TODO: qbn 상황에서는 전자를 고르는거고.
                # TODO: 반대로, 위의 2번같이 가까운 qx_qc를 골랐을 때, logit이 굉장히 낮은 곳으로 가라는 선택이 나올수도 있다.
                # 즉 양쪽 다, deterministic action 이어야 의미가 있음.
                if actual[0][0] != q_action:
                    print('logit:{}, action:{}, state_desc_action:{}'.format(torch.exp(action_out[0][0][0]), actual[0][0], q_action ))
                actual[0][0] = q_action

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

        # BK
        # if eval_mode:
            # self.writer.add_scalar('eval_mode'+'/rewards per episode', np.mean(stat['reward']), num_episodes)
            # self.writer.add_scalar("eval_mode"+"/agent0's reward per episode", stat['reward'][0], num_episodes)

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

    def _get_index(self, source, item, force=True):
        """
        Returns index of the item in the source

        Parameters
        ----------
        source: np-array comprising of unique elements (set)
        item: target item(array)
        force: if True: In case item not found; it will add the item and return the corresponding index

        Returns
        -------
        source: update된 state/obs/comm space
        _index: int
                if new_observation: _index = len(source) - 1
                else:               _index = 해당 obs의 기존 index
        
        """
        _index = np.where(np.all(source==item, axis=1))[0] if len(source) != 0 else []
        if len(_index) != 0:  # already in before_observation
            _index = _index[0]
        elif force:  # new_observation
            source = source.tolist()
            source.append(item)
            source = np.array(source)
            _index = len(source) - 1
        else:
            _index = None
        
        return source, _index