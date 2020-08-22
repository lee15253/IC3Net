import os
import torch
import random
import scipy.misc
import numpy as np
import logging, sys
from collections import deque
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.autograd import Variable
# from tools import ensure_directory_exits
from PIL import Image, ImageFont, ImageDraw
# TODO: 패키지 정리
from models import MMNet
from qbn_trainer import QBNTrainer
import ipdb
import itertools


class MooreMachine():
    """
    1. Generating Finite State Machine from trained quantized-MMnet
    2. Minimizing / Pruning

    Attributes
    -----------
    self.transaction : {int: {int: int}} ({h_t: {o_t: h_t+1}})
                        transaction dictionary
    self.state_desc  : {int: {'action':int, 'description': np.array}} ({state_index: {'action': action_index, 'description': state}})
                        state-action dictionary
    self.obs_space   : [[np.array]] ([[obs]])
                        observation array
    self.net         :  quantized-bottlneck inserted RL network
    
    Methods
    -------
    generate_fsm (~~)
    ~~~~~~~~~~~ (~~)
    """

    def __init__(self, args, env, obs_qb_net, comm_qb_net, hidden_qb_net, 
                policy_net, mmn_directory, storage, writer, t={}, sd={}, ss=np.array([]), 
                os=np.array([]), cs=np.array([]), start_state=0, total_actions=None):
        
        self.args = args
        self.transaction = t
        self.state_desc = sd
        self.state_space = ss
        self.obs_space = os
        self.comm_space = cs
        self.start_state = start_state
        self.minimized = False
        self.obs_minobs_map = None
        self.minobs_obs_map = None
        self.frequency = None
        self.trajectory = None
        self.total_actions = total_actions
        self.storage = storage
        self.writer = writer
        self.model = MMNet(policy_net, obs_qb_net, comm_qb_net, hidden_qb_net)
        ipdb.set_trace()

        q_parameter = torch.load(mmn_directory+'/mmn.pth')  # fine-tuned (final) parameter
        obs_paratmeter = {k.split("obs_qb_net.")[1]: v for k,v in q_parameter.items() if k.startswith("obs")}
        self.model.obs_qb_net.load_state_dict(obs_paratmeter)  # load obs_qb_net
        hidden_paratmeter = {k.split("hidden_qb_net.")[1]: v for k,v in q_parameter.items() if k.startswith("hidden")}
        self.model.hidden_qb_net.load_state_dict(hidden_paratmeter)  # load hidden_qb_net
        comm_paratmeter = {k.split("comm_qb_net.")[1]: v for k,v in q_parameter.items() if k.startswith("comm")}
        self.model.comm_qb_net.load_state_dict(comm_paratmeter)  # load comm_qb_net
        policy_paratmeter = {k.split("policy_net.")[1]: v for k,v in q_parameter.items() if k.startswith("policy")}
        self.model.policy_net.load_state_dict(policy_paratmeter)  # load policy_net

        # initialize qbn_trainer with fine-tuned paramaters
        self.qbn_trainer = QBNTrainer(args, env, self.model.policy_net, self.model.obs_qb_net, 
                                 self.model.comm_qb_net, self.model.hidden_qb_net, self.storage, self.writer)

    def make_fsm(self, episodes=10, seed=1):
        """
        Makes FSM.
        1. rollout with fine-tuned RL agents -> get final (q_obs, q_hx, q_comm)
        2. (q_obs, q_hx, q_comm) -> makes FSM (transaction table, obs/comm - hx mapping)

        Parameters
        ----------
        episodes: int
                  # of rollout epoch
        seed:     int

        Return
        ------
        self.transaction
        self.state_desc
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        ipdb.set_trace()
        
        self.model.eval()
        
        # TODO: how many rollout_steps are needed?
        # n_rollout_steps = self.args.num_train_rollout_steps
        n_rollout_steps = episodes
        self.qbn_trainer.perform_rollouts(self.model, n_rollout_steps,
                                           net_type = 'fine_tuned_model(before_FSM)', store = True)
        q_x_batch, q_c_batch, q_h_batch, a_batch = self.qbn_trainer.storage.fetch_fsm_data()                                  
        new_entries = self.update_fsm(q_x_batch, q_c_batch, q_h_batch, a_batch)

    def update_fsm(self, q_x_batch, q_c_batch, q_h_batch, a_batch):
        """
        udpate self.transaction, self.state_desc
        
        parameters
        ----------
        self.state_desc  : {int: {'action':int, 'description': np.array}} ({state_index: {'action': action_index, 'description': state}})
                        state-action dictionary
        self.transaction : {int: {int: int}} ({h_t: {o_t: h_t+1}})
                        transaction dictionary

        return
        ------
        new_entries
        
        """
        # (q_x[t] + q_c[t], hidden_t, cell_t)  ->  LSTM  ->  q_h[t]  ->  a_t[t]
        new_entries = [] 
        self.second_state = set()

        # TODO: 어거지로 start_space(0000)을 고려해줌
        # start_state = self.storage.h_t_batch[0]
        # q_start_state = self.model.hidden_qb_net(start_state)
        # _, self.start_state = self._get_index(self.state_space, q_start_state, force=True)
        # TODO: action도 구해서 desc, transaction 다 update?

        for t in range(len(q_x_batch)-1):
            self.obs_space, qx_index = self._get_index(self.obs_space, q_x_batch[t])
            self.comm_space, qc_index = self._get_index(self.comm_space, q_c_batch[t])
            self.state_space, qh_index = self._get_index(self.state_space, q_h_batch[t])
            self.state_space, qh_t1_index = self._get_index(self.state_space, q_h_batch[t+1]) 

            # update self.state_desc
            # initialize self.state_desc[qh_index]
            if qh_index not in self.state_desc:
                self.state_desc[qh_index] = {'action': str(a_batch[t]), 'description': q_h_batch[0]}

            # TODO: When? Maybe partial-fsm을 처리할 때?
            if self.state_desc[qh_index]['action'] == str(None) and a_batch[t] is not None:
                ipdb.set_trace()
                self.state_desc[qh_index]['action'] = str(a_batch[t])
            
            # update self.transaction
            qx_qc = str(qx_index)+'_'+str(qc_index)
            # initialize self.transaction[qh_index]
            if qh_index not in self.transaction:  
                # TODO: 코드 바꿔서 partial이나 minimization에서 문제생길수도
                # self.transaction[qh_index] = {str(i)+'_'+str(j): None for i in range(len(self.obs_space)) for j in range(len(self.comm_space))}
                # new_entries += [(s_i, str(i)+'_'+str(j)) for i in range(len(self.obs_space)) for j in range(len(self.comm_space))]
                self.transaction[qh_index] = {qx_qc: None}
            # self.transaction[qh_index] exists & obs/comm are not in it
            elif qx_qc not in self.transaction[qh_index]:
                self.transaction[qh_index][qx_qc] = None  # TODO: 코드 바꿔서 partial에서 문제생길수도
            
            new_entries += [(qh_index, qx_qc)]
            self.transaction[qh_index][qx_qc] = qh_t1_index
            
            ## TODO: qh_index[0]이 [0,0,...,0]이 아니라 그 다음 거고, initial observation도 조금씩 달라서,
            # 시작 state가 동일하지 않다. -> 문제가 되려나?
            # 일단 q_h_1의 모음 (env.reset() -> h_0(=0벡터) -> LSTM -> h_1)
            if (t % self.args.max_steps) == 0:
                self.second_state.add(qh_index)

        return new_entries

    
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
        _index: int
                if new_observation: _index = len(source) - 1
                else:               _index = 해당 obs의 기존 index
        ~~
        """
        # ipdb.set_trace()
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

    def save(self, info_file):
        # ipdb.set_trace()
        info_file.write('Total Unique States:{}\n'.format(len(self.state_space)))
        info_file.write('Total Unique Obs:{}\n'.format(len(self.obs_space)))
        info_file.write('Total Unique Comm:{}\n'.format(len(self.comm_space)))
        info_file.write('Start h_t_1:{}\n'.format(self.second_state))

        # ht - at mapping table
        info_file.write('\n\nStates Description:\n')
        t1 = PrettyTable(["Name", "Action", "Description" if not self.minimized else 'Sub States'])
        for k in sorted(self.state_desc.keys()):
            _state_info = self.state_desc[k]['description' if not self.minimized else 'sub_states']
            t1.add_row([k, self.state_desc[k]['action'], _state_info])
        info_file.write(t1.__str__() + '\n')

        # transaction table
        if not self.minimized:
            qc = [list(v.keys()) for i,v in self.transaction.items()]
            qc = list(itertools.chain.from_iterable(qc))
            column_names = [""] + qc
            column_names = list(set(column_names))
            ipdb.set_trace()
            t = PrettyTable(column_names)
            for key in sorted(self.transaction.keys()):
                t.add_row([key]+[(self.transaction[key][c] if c in self.transaction[key] else None) for c in column_names[1:]])

        info_file.write('\n\nTransaction Matrix:    (StateIndex_ObservationIndex x StateIndex)' + '\n')
        info_file.write(t.__str__())


        info_file.close()