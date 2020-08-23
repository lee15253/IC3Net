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
from tqdm import tqdm

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

    def make_fsm(self, num_rollout_steps=100, seed=1):
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
        
        self.model.eval()
        self.qbn_trainer.perform_rollouts(self.model, num_rollout_steps,
                                           net_type = 'fine_tuned_model(before_FSM)', store = True)
        print('\nrollout finished')
        q_x_batch, q_c_batch, q_h_batch, a_batch = self.qbn_trainer.storage.fetch_fsm_data()
        new_entries = self.update_fsm(q_x_batch, q_c_batch, q_h_batch, a_batch)
        print('self.transaction, self.state_desc initialized')

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
        qc = [list(v.keys()) for i,v in self.transaction.items()]
        qc = list(itertools.chain.from_iterable(qc))

        if not self.minimized:
            info_file.write('Total Unique Obs:{}\n'.format(len(self.obs_space)))
            info_file.write('Total Unique Comm:{}\n'.format(len(self.comm_space)))
            info_file.write('Total Unique Obs-comm:{}\n'.format(len(qc)))
        else: # TODO:
            temp = list(map(lambda x: x.split('_'), list(self.minobs_obs_map.keys())))
            temp = list(zip(*temp))
            num_obs, num_comm = len(set(temp[0])), len(set(temp[1]))
            info_file.write('Total Unique Obs-comm:{}\n'.format(len(self.minobs_obs_map.keys())))
            info_file.write('Total Unique Obs:{}\n'.format(num_obs))
            info_file.write('Total Unique Comm:{}\n'.format(num_comm))

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
            column_names = [""] + qc
            column_names = list(set(column_names))
            t = PrettyTable(column_names)
            for key in sorted(self.transaction.keys()):
                t.add_row([key]+[(self.transaction[key][c] if c in self.transaction[key] else None) for c in column_names[1:]])
        else:
            ipdb.set_trace()
            column_names = [""] + sorted(self.transaction[list(self.transaction.keys())[0]].keys())
            t = PrettyTable(column_names)
            for key in sorted(self.transaction.keys()):
                t.add_row([key] + [self.transaction[key][c] for c in column_names[1:]])

        info_file.write('\n\nTransaction Matrix:    (StateIndex_ObservationIndex x StateIndex)' + '\n')
        info_file.write(t.__str__())
        # info_file.write('\n\nTransaction Matrix:    (StateIndex_ObservationIndex x StateIndex)' + '\n')
        # info_file.write(t.__str__())
        info_file.close()

        

    def minimize_partial_fsm(self):
        """
        Minimizing the whole Finite State Machine(FSM) to fewer states.
        """
        # Considering communications        
        qc_list = [list(v.keys()) for i,v in self.transaction.items()]
        qc_list = list(itertools.chain.from_iterable(qc_list))

        _states = sorted(self.transaction.keys())
        compatibility_mat = {s: {p: False if self.state_desc[s]['action'] != self.state_desc[p]['action'] else None
                                 for p in _states[:i + 1]}
                             for i, s in enumerate(_states[1:])}
        unknowns = []
        for s in compatibility_mat.keys():
            for k in compatibility_mat[s].keys():
                if compatibility_mat[s][k] is None:
                    unknowns.append((s, k))

        ipdb.set_trace()
        unknown_lengths = deque(maxlen=100000)

        pbar = tqdm(total=len(unknowns))

        while len(unknowns) != 0:
            # next 3 lines are experimental
            pbar.update(1)
            if len(unknown_lengths) > 0 and unknown_lengths.count(unknown_lengths[0]) == unknown_lengths.maxlen:
                s, k = unknowns[-1]
                compatibility_mat[s][k] = True

            s, k = unknowns.pop(0)
            if compatibility_mat[s][k] is None:
                compatibility_mat[s][k] = []
                for qc_i in qc_list:
                    if (qc_i not in self.transaction[s]) or (self.transaction[s][qc_i] is None) or \
                            (qc_i not in self.transaction[k]) or (self.transaction[k][qc_i] is None):
                        pass
                    else:
                        next_s, next_k = self.transaction[s][qc_i], self.transaction[k][qc_i]
                        action_next_s = self.state_desc[next_s]['action']
                        action_next_k = self.state_desc[next_k]['action']
                        # if next_s != next_k and next_k != k and next_s != s:
                        if next_s != next_k and not (next_k == k and next_s == s):
                            if action_next_s != action_next_k:
                                compatibility_mat[s][k] = False
                                break
                            first, sec = sorted([next_k, next_s])[::-1]
                            if type(compatibility_mat[first][sec]).__name__ == 'bool' and not \
                                    compatibility_mat[first][sec]:
                                compatibility_mat[s][k] = False
                                break
                            elif compatibility_mat[first][sec] is None or \
                                    type(compatibility_mat[first][sec]).__name__ != 'bool':
                                compatibility_mat[s][k].append((first, sec))

            elif type(compatibility_mat[s][k]).__name__ != 'bool':
                for i, (m, n) in enumerate(compatibility_mat[s][k]):
                    if type(compatibility_mat[m][n]).__name__ == 'bool' and not compatibility_mat[m][n]:
                        compatibility_mat[s][k] = False
                        break
                    elif type(compatibility_mat[m][n]).__name__ == 'bool' and compatibility_mat[m][n]:
                        compatibility_mat[s][k].pop(i)

            if type(compatibility_mat[s][k]).__name__ != 'bool':
                if len(compatibility_mat[s][k]) == 0:
                    compatibility_mat[s][k] = True
                else:
                    unknowns.append((s, k))

            unknown_lengths.append(len(unknowns))
            # print('처리해야할 (남은) unknowns 개수:',len(unknowns))
        pbar.close()

        # hidden_state가 커서 여기까지 2분 이상 소요됨
        # new_states : 새로운 minimized hideen state.
        # ipdb.set_trace()
        new_states = []
        new_state_info = {}
        processed = {x: False for x in _states}
        belongs_to = {_: None for _ in _states}
        # 모든 state를 돌면서, 1. comp_pair(?)를 구하고 2. 걔랑 compatible한 것들을 self.traverse_compatible_states로 찾아서
        # _new_state에 subgroup으로 묶이게 한다. 이때 한 state가 여러 new_state에 포함된다. 
        # 구분 기준을 잘 모르겠으나, 이후 transaction을 할 때 같은 동작 방식인것들로 추정.
        for s in sorted(_states):
            if not processed[s]:
                comp_pair = [sorted((s, x))[::-1] for x in _states if
                             (x != s and compatibility_mat[max(s, x)][min(s, x)])]
                if len(comp_pair) != 0:
                    _new_state = self.traverse_compatible_states(comp_pair, compatibility_mat)
                    _new_state.sort(key=len, reverse=True)
                else:
                    _new_state = [[s]]
                for d in _new_state[0]:
                    processed[d] = True
                    belongs_to[d] = len(new_states)
                new_state_info[len(new_states)] = {'action': self.state_desc[_new_state[0][0]]['action'],
                                                   'sub_states': _new_state[0]}
                new_states.append(_new_state[0])

        new_trans = {}
        for i, s in enumerate(new_states):
            new_trans[i] = {}
            try:
                for qc_i in qc_list:
                    new_trans[i][qc_i] = None
                    for sub_s in s:
                        if qc_i in self.transaction[sub_s] and self.transaction[sub_s][qc_i] is not None:
                            new_trans[i][qc_i] = belongs_to[self.transaction[sub_s][qc_i]]
                            break
            except:
                ipdb.set_trace()

        # if the new_state comprising of start-state has just one sub-state ;
        # then we can merge this new_state with other new_states as the action of the start-state doesn't matter
        # TODO: 일단 start_state 관련 뭔소린지 모르겠으므로 일단 생략조진다
        # start_state_p = belongs_to[self.start_state]
        # if len(new_states[start_state_p]) == 1:
        #     start_state_trans = new_trans[start_state_p]
        #     for state in new_trans.keys():
        #         if state != start_state_p and new_trans[state] == start_state_trans:
        #             new_trans.pop(start_state_p)
        #             new_state_info.pop(start_state_p)
        #             new_state_info[state]['sub_states'] += new_states[start_state_p]

        #             # This could be highly wrong (On God's Grace :D )
        #             for _state in new_trans.keys():
        #                 for _o in new_trans[_state].keys():
        #                     if new_trans[_state][_o] == start_state_p:
        #                         new_trans[_state][_o] = state

        #             start_state_p = state
        #             break

        # TODO: Pong의 경우, S_2에 O1이 들어오면 어케하는거? -> O_1이라는건 어떤 obs들의 집합이고, S_2일때 해당 observation이 들어오는
        # 경우가 없었다는것 -> 맞나? -> 근데 순서가 이상함
        # 이호준: Pong 그림: S2의 상태에선, O_1이 들어올 일이 없는 것이기 때문에 O_1이 없다. 
        # --> 그 extract_from_nn의 if not partial 하면, 그 unknown에 대해서 다 forwarding하면서 check해주는듯

        # Minimize Observation Space (Combine observations which show the same transaction behaviour for all states)
        ipdb.set_trace()

        _obs_minobs_map = {}
        _minobs_obs_map = {}
        _trans_minobs_map = {}
        min_trans = {s: {} for s in new_trans.keys()}
        # new_qc_i = 0
        for qc_i in qc_list:
            _trans_key = [new_trans[s][qc_i] for s in sorted(new_trans.keys())].__str__()
            if _trans_key not in _trans_minobs_map:
                # new_qc_i += 1  # TODO: qc버전으로 수정해야함!!!!!!!
                o = qc_i
                _trans_minobs_map[_trans_key] = o
                _minobs_obs_map[o] = [qc_i]
                for s in new_trans.keys():
                    min_trans[s][o] = new_trans[s][qc_i]
            else:
                _minobs_obs_map[_trans_minobs_map[_trans_key]].append(qc_i)
            _obs_minobs_map[qc_i] = _trans_minobs_map[_trans_key]

        # Update information
        self.transaction = min_trans
        self.state_desc = new_state_info
        self.state_space = list(self.transaction.keys())
        # self.start_state = start_state_p
        self.obs_minobs_map = _obs_minobs_map
        self.minobs_obs_map = _minobs_obs_map
        self.minimized = True

    @staticmethod
    def traverse_compatible_states(states, compatibility_mat):
        for i, s in enumerate(states):
            for j, s_next in enumerate(states[i + 1:]):
                compatible = True
                for m in s:
                    for n in s_next:
                        if m != n and not compatibility_mat[max(m, n)][min(m, n)]:
                            compatible = False
                            break
                    if not compatible:
                        break
                if compatible:
                    _states = states[:i] + [sorted(list(set(s + s_next)))] + states[i + j + 2:]
                    return MooreMachine.traverse_compatible_states(_states, compatibility_mat)
        return states