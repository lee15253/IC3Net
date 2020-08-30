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
import pickle

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

    """

    def __init__(self, args, env, obs_qb_net, comm_qb_net, hidden_qb_net, 
                policy_net, mmn_directory, storage, writer, t={}, sd={}, ss=np.array([]), 
                os=np.array([]), cs=np.array([]), start_state=0, total_actions=None):
        
        self.args = args
        self.env = env
        self.transaction = t
        self.state_desc = sd
        self.state_space = ss
        self.obs_space = os
        self.comm_space = cs
        self.start_state = start_state
        self.mmn_directory = mmn_directory
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

    def make_fsm(self,num_rollout_steps=10, seed=1, saved=False):
        """
        Makes FSM.
        1. rollout with fine-tuned RL agents -> get final (q_obs, q_hx, q_comm)
        2. (q_obs, q_hx, q_comm) -> makes FSM (transaction table, obs/comm - hx mapping)

        Parameters
        ----------
        episodes: int
                  # of rollout epoch
        seed:     int

        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.model.eval()
        self.qbn_trainer.perform_rollouts(self.model, num_rollout_steps,
                                           net_type = 'fine_tuned_model(before_FSM)', store = True)
        print('\nrollout finished')

        q_x_batch, q_c_batch, q_ht1_batch, a_t1_batch, curr_t = self.qbn_trainer.storage.fetch_fsm_data()

        # # For debugging (내 방식이 맞는지)
        # 1. make_fsm에 save argument 추가해서 2. 먼저 원본을 saved=False해서 돌리고 3. 내 버전을 saved=True해서 돌려서 minimized 같은지 확인
        # if saved == False:
        #     q_x_batch, q_c_batch, q_ht1_batch, a_t1_batch, curr_t = self.qbn_trainer.storage.fetch_fsm_data()
            # batch_files = [q_x_batch, q_c_batch, q_ht1_batch, a_t1_batch, curr_t]
            # with open(os.path.join(self.mmn_directory,'batch_files.p'), 'wb') as handle:
            #     pickle.dump(batch_files, handle, protocol = pickle.HIGHEST_PROTOCOL)
            # print('saved!')
        # else:
            # print('load!')
            # with open(os.path.join(self.mmn_directory, 'batch_files.p'), 'rb') as handle:
            #     b_files = pickle.load(handle)
            # q_x_batch, q_c_batch, q_ht1_batch, a_t1_batch, curr_t = b_files        

        new_entries = self.update_fsm(q_x_batch, q_c_batch, q_ht1_batch, a_t1_batch, curr_t)
        print('\n FSM updated')

    def update_fsm(self, q_x_batch, q_c_batch, q_ht1_batch, a_t1_batch, curr_t):
        """
        udpate self.transaction, self.state_desc from rollout_storage
        
        parameters
        ----------
        self.state_desc  : {int: {'action':int, 'description': np.array}} ({state_index: {'action': action_index, 'description': state}})
                        state-action dictionary
        self.transaction : {int: {int: int}} ({h_t: {o_t: h_t+1}})
                        transaction dictionary

        return
        ------
        new_entries  : FIXME: Not used.
        
        """
        # (q_x[t] + q_c[t], hidden_t, cell_t)  ->  LSTM  ->  q_h[t+1]  ->  a_t[t+1]
        new_entries = [] 
        total_qx_qc = set()

        game_length = self.args.max_steps
        for t in range(len(q_x_batch)):
            self.obs_space, qx_index = self._get_index(self.obs_space, q_x_batch[t])
            self.comm_space, qc_index = self._get_index(self.comm_space, q_c_batch[t])

            # t=0에서는 input_h를 사용하고 그 외는 output_h를 사용해야 한다.
            if int(curr_t[t]) == 0:
                # input h를 quantize (same as hidden.encode(000000000))
                first_h = self.model.hidden_qb_net.encode(torch.zeros(self.args.hid_size)).detach()
                self.state_space, qh_index = self._get_index(self.state_space, first_h.numpy())
                self.state_space, qh_t1_index = self._get_index(self.state_space, q_ht1_batch[t])  # output h를 quantize 
                if t==0:
                    self.start_state = qh_index
            else:
                self.state_space, qh_index = self._get_index(self.state_space, q_ht1_batch[t-1])
                self.state_space, qh_t1_index = self._get_index(self.state_space, q_ht1_batch[t]) 


            # update self.state_desc
            if int(curr_t[t]) == 0:  # extract (h0 -> action) 
                decoded_qh = self.model.hidden_qb_net.decode(first_h)
                decoded_qh = decoded_qh.view(1, self.args.hid_size)
                prob = [F.softmax(head(decoded_qh), dim=-1) for head in self.model.policy_net.heads][0]
                action = torch.argmax(prob).item()  # argmax action selection
                self.state_desc[qh_index] = {'action': str(action), 'description': first_h.numpy()}
                assert qh_index == 0, 'error in quantized_state indexing'
            # initialize self.state_desc[qh_index] ((h1,a1), ..., (h20,a20))
            if qh_t1_index not in self.state_desc:  
                action = np.argmax(a_t1_batch[t])
                self.state_desc[qh_t1_index] = {'action': str(action), 'description': q_ht1_batch[t]}
    

            # update self.transaction
            qx_qc = str(qx_index)+'_'+str(qc_index)
            
            # 1. 원본 방식 !!!
            # initialize self.transaction[qh_index]
            # total_qx_qc.add(qx_qc)
            # if qh_index not in self.transaction:  
            #     self.transaction[qh_index] = {i: None for i in total_qx_qc}
            #     # new_entries += [(s_i, str(i)+'_'+str(j)) for i in range(len(self.obs_space)) for j in range(len(self.comm_space))]
            #     # self.transaction[qh_index] = {qx_qc: None}  FIXME: minimize시, 내 코드대로 하면 걍 상관없는것까지 compat_mat이 True가 되버리는 것 같다. 내 코드 잠시 off
            # elif qx_qc not in self.transaction[qh_index]:
            #     # self.transaction[qh_index][qx_qc] = None  FIXME: 내 코드 잠시 off
            #     for i in total_qx_qc:
            #         if i not in self.transaction[qh_index]:
            #             self.transaction[qh_index][i] = None
            #             # if str(i)+'_'+str(j) != qx_qc:
            #             #     new_entries.append((qh_t1_index, str(i)+'_'+str(j)))

            # if qh_t1_index not in self.transaction:
            #     self.transaction[qh_t1_index] = {i: None for i in total_qx_qc}
            # elif qx_qc not in self.transaction[qh_t1_index]:
            #     for i in total_qx_qc:
            #         if i not in self.transaction[qh_t1_index]:
            #             self.transaction[qh_t1_index][i] = None

            
            # 2. 내 방식 !!!
            if qh_index not in self.transaction:
                self.transaction[qh_index] = {qx_qc: None}
            elif qx_qc not in self.transaction[qh_index]:
                self.transaction[qh_index][qx_qc] = None 
            # t=39 일때, qh_index=36, qh_t1_index=37이라고 하면, t=40때 qh_index=0, qh_t1_index = 38이 되어
            # self.transaction[38]이 없게 된다. -> {}으로 일단 initialize하고 밑에 minimize 시 unknown에서 처리되길 기대함.
            # 근본적인 문제는 h_x가 경우의 수가 너무 많아서, qh_index=38을 만날 기회가 다시는 없기 때문에 error가 나는것.
            # 밑처럼 하면 self.transaction = {0:{'0_0':1, '13_31':3}, ...,  16:{},  17:{'5_6':18}, ...}
            if qh_t1_index not in self.transaction: 
                if t+1 == len(q_x_batch):
                    self.transaction[qh_t1_index] = {}
                else:
                    if curr_t[t+1] == 0:
                        self.transaction[qh_t1_index] = {}

            self.transaction[qh_index][qx_qc] = qh_t1_index

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

    def save(self, info_file):
        """
        Save FSM / minimized_FSM.txt

        Parameters
        ----------
        info: empty text file         
        """
        assert len(self.state_desc) == len(self.transaction), 'len(self.state_desc) != len(self.transaction)'
        info_file.write('Total Unique States:{}\n'.format(len(self.state_space)))
        qc = [list(v.keys()) for i,v in self.transaction.items()]
        qc = list(itertools.chain.from_iterable(qc))
        qc = list(set(qc))

        if not self.minimized:
            info_file.write('Total Unique Obs-comm 조합:{}\n'.format(len(qc)))
            info_file.write('Total Unique Obs:{}\n'.format(len(self.obs_space)))
            info_file.write('Total Unique Comm:{}\n'.format(len(self.comm_space)))
            
        else: 
            temp = list(map(lambda x: x.split('_'), list(self.minobs_obs_map.keys())))
            temp = list(zip(*temp))
            num_obs, num_comm = len(set(temp[0])), len(set(temp[1]))
            info_file.write('Total Unique Obs-comm 조합:{}\n'.format(len(self.minobs_obs_map.keys())))
            info_file.write('Total Unique Obs:{}\n'.format(num_obs))
            info_file.write('Total Unique Comm:{}\n'.format(num_comm))

        info_file.write('Start state:{}\n'.format(self.start_state))

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
            # TODO: self.transaction.keys()가 어케생긴거였지? 왜 0번 index거를 가져오지
            column_names = [""] + sorted(self.transaction[list(self.transaction.keys())[0]].keys())
            t = PrettyTable(column_names)
            for key in sorted(self.transaction.keys()):
                t.add_row([key] + [self.transaction[key][c] for c in column_names[1:]])
            # unminimized일 경우 너무 커서 안써진다. minimized 버전일경우만 write하자.
            info_file.write('\n\nTransaction Matrix:    (StateIndex_ObservationIndex x StateIndex)' + '\n')
            info_file.write(t.__str__())

        info_file.close()

    def minimize_partial_fsm(self):
        """
        Minimizing the whole Finite State Machine(FSM) to fewer states.
        """
        # Considering communications        
        qc = [list(v.keys()) for i,v in self.transaction.items()]
        qc = list(itertools.chain.from_iterable(qc))
        qc_list = list(set(qc))

        _states = sorted(self.transaction.keys())
        # fisrt_state(_state[0])을 제외하고, 돌아가면서 본인 이전(내가 state 10이라면, 0~9까지 p로 돌면서)과
        # action이 다를 경우, compatibility_map[s][p] = False, 같을 경우 None으로 추가한다.
        # compatibility_mat[10]
        # {0: False, 1: False, 2: False, 3: False, 4: False, 5: None, 6: None, 7: None, 8: None, 9: None}
        compatibility_mat = {s: {p: False if self.state_desc[s]['action'] != self.state_desc[p]['action'] else None
                                 for p in _states[:i + 1]}
                             for i, s in enumerate(_states[1:])}
        unknowns = []
        # None을 전부 모은다.
        for s in compatibility_mat.keys():
            for k in compatibility_mat[s].keys():
                if compatibility_mat[s][k] is None:
                    unknowns.append((s, k))
        unknown_lengths = deque(maxlen=100000)

        pbar = tqdm(total=len(unknowns))

        # compatibility_matrix의 None을 True/False로 채워나간다.
        while len(unknowns) != 0:
            pbar.update(1)
            # TODO: next 3 lines are experimental  
            if len(unknown_lengths) > 0 and unknown_lengths.count(unknown_lengths[0]) == unknown_lengths.maxlen:
                ipdb.set_trace()
                s, k = unknowns[-1]
                compatibility_mat[s][k] = True

            s, k = unknowns.pop(0)  # 앞 쪽 state부터 check한다
            # 1. compatibility_mat[s][k]가 None인 경우 -> False로 바꾸던가, (next_s, next_k)를 append한다
            if compatibility_mat[s][k] is None:
                compatibility_mat[s][k] = []
                # For every obs_comm(=input)
                for qc_i in qc_list:
                    # 이 중 하나라도 틀리면 같은 state가 아니다
                    if (qc_i not in self.transaction[s]) or (self.transaction[s][qc_i] is None) or \
                            (qc_i not in self.transaction[k]) or (self.transaction[k][qc_i] is None):
                        pass
                    else:
                        next_s, next_k = self.transaction[s][qc_i], self.transaction[k][qc_i]
                        action_next_s = self.state_desc[next_s]['action']
                        action_next_k = self.state_desc[next_k]['action']
                        # TODO: 이렇게하면, obs-comm조합에 대해서는 맞춰진다. (저 둘은 같은 state라고 생각하는것)
                        # 다만, comm, obs 각각의 개수가 조금 달라진다.
                        if next_s == k and next_k == s:
                            print('next_s==k, next_k==s -> True처리, s,k={}'.format(s,k))
                            compatibility_mat[s][k] = True

                        if next_s != next_k and not (next_k == k and next_s == s):
                            # next_state에서의 action이 다른 경우 out
                            if action_next_s != action_next_k:
                                compatibility_mat[s][k] = False
                                break
                            first, sec = sorted([next_k, next_s])[::-1]
                            
                            if type(compatibility_mat[first][sec]).__name__ == 'bool' and not compatibility_mat[first][sec]:
                                compatibility_mat[s][k] = False
                                break
                            # compatibility_mat[next_s][next_k]가 None이나 list인 경우 compatibility_mat[s][k]에 (next_s, next_k) append
                            elif compatibility_mat[first][sec] is None or type(compatibility_mat[first][sec]).__name__ != 'bool':
                                compatibility_mat[s][k].append((first, sec))
                            

            # 2. list인 경우 -> 여기 들어있는 m,n은 next_s, next_k 
            elif type(compatibility_mat[s][k]).__name__ != 'bool':
                for i, (m, n) in enumerate(compatibility_mat[s][k]):
                    # compatibiltiy_mat[next_s][next_k]가 False면 얘도 False
                    if type(compatibility_mat[m][n]).__name__ == 'bool' and not compatibility_mat[m][n]:
                        compatibility_mat[s][k] = False
                        break
                    # compatibiltiy_mat[next_s][next_k]가 True라면 그놈을 없애버린다.
                    elif type(compatibility_mat[m][n]).__name__ == 'bool' and compatibility_mat[m][n]:
                        compatibility_mat[s][k].pop(i)
                    # compatibiltiy_mat[next_s][next_k]가 list라면 처리를 안해주는듯 하다

            if type(compatibility_mat[s][k]).__name__ != 'bool':
                # 1. s,k에서 action이 같고 2.next_s, next_k에서의 action도 같고
                # 3. next_s, next_k들의 compatibility_mat이 True라면 -> True이다.
                # 혹은 !! 저 위의 if ~ or ~ or ~ or ~에 전부 걸려서 그냥 []인채로 끝나는 애들도, True로 mapping된다.
                # TODO: 이유는 모르겠으나, 원본 코드도 똑같이 동작한다.
                if len(compatibility_mat[s][k]) == 0:
                    compatibility_mat[s][k] = True
                else:
                    unknowns.append((s, k))

            unknown_lengths.append(len(unknowns))
            # print('처리해야할 (남은) unknowns 개수:',len(unknowns))
        pbar.close()

        # new_states : 새로운 minimized hideen state.
        new_states = []
        new_state_info = {}
        processed = {x: False for x in _states}
        belongs_to = {_: None for _ in _states}
        # 모든 state를 돌면서, 1. comp_pair(?)를 구하고 2. 걔랑 compatible한 것들을 self.traverse_compatible_states로 찾아서
        # _new_state에 subgroup으로 묶이게 한다. 이때 한 state가 여러 new_state에 포함된다. 
        # 구분 기준을 잘 모르겠으나, 이후 transaction을 할 때 같은 동작 방식인것들로 추정.
        # !!! 옛날에 잘못생각함. 한 state가 여러 new_state에 안들어감! (len(self.state_desc) = len(new_states[0~3]))
        # TODO: ㅅㅂ!! -> 아니다. 안겹친다 (MMN 원본도그랬음). ex) state 0과 state4가 딱 state한개만 다름
        # --> transaction table 에서 딱 한 column만 달라지게 됨
        for s in sorted(_states):
            if not processed[s]:
                # compatibility_mat이 True인 쌍을 다 묶는다
                comp_pair = [sorted((s, x))[::-1] for x in _states if
                             (x != s and compatibility_mat[max(s, x)][min(s, x)])]
                # 확인한 바로는 comp_pair에서 같이 묶인 것들은 다 같은 state로 묶이는 것 같음. ㅇㅇ
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

        # 같은 sub_state로 묶인것들을 돌아가면서, 그 transaction을 전부 new_state의 transaction으로 갖다박는다.
        # 이래서 decision point가 많아지는구만
        new_trans = {}
        for i, s in enumerate(new_states):
            new_trans[i] = {}
            for qc_i in qc_list:
                new_trans[i][qc_i] = None
                for sub_s in s:
                    if qc_i in self.transaction[sub_s] and self.transaction[sub_s][qc_i] is not None:
                        new_trans[i][qc_i] = belongs_to[self.transaction[sub_s][qc_i]]
                        break

        # if the new_state comprising of start-state has just one sub-state ;
        # then we can merge this new_state with other new_states as the action of the start-state doesn't matter
        start_state_p = belongs_to[self.start_state]
        if len(new_states[start_state_p]) == 1:
            start_state_trans = new_trans[start_state_p]
            for state in new_trans.keys():
                if state != start_state_p and new_trans[state] == start_state_trans:
                    print('start_state is merged to another state')
                    print('original_state: {}, new_state: {}'.format(start_state_p, state))
                    new_trans.pop(start_state_p)
                    new_state_info.pop(start_state_p)
                    new_state_info[state]['sub_states'] += new_states[start_state_p]

                    # This could be highly wrong (On God's Grace :D )
                    for _state in new_trans.keys():
                        for _o in new_trans[_state].keys():
                            if new_trans[_state][_o] == start_state_p:
                                new_trans[_state][_o] = state

                    start_state_p = state
                    break

        # TODO: Pong의 경우, S_2에 O1이 들어오면 어케하는거? -> O_1이라는건 어떤 obs들의 집합이고, S_2일때 해당 observation이 들어오는
        # 경우가 없었다는것 -> 맞나? -> 근데 순서가 이상함
        # 이호준: Pong 그림: S2의 상태에선, O_1이 들어올 일이 없는 것이기 때문에 O_1이 없다. --> 맞음 !
        # --> 그 extract_from_nn의 if not partial 하면, 그 unknown에 대해서 다 forwarding하면서 check해주는듯

        # Minimize Observation Space (Combine observations which show the same transaction behaviour for all states)
        # ex) (S1,S2,S3,S4,S5) -> O13 -> (S1',S2',S3',S4',S5'), (S1,S2,S3,S4,S5) -> O21 -> (S1',S2',S3',S4',S5')
        # Then O13, O21을 merge
        _obs_minobs_map = {}  # {'1_1':'1_1',  '10_13':'10_13',  '21_35':'1_1'(1_1로merge됨)}
        _minobs_obs_map = {}  # {'1_1':['1_1', '21_35'], '10_13':[10_13,...]}
        _trans_minobs_map = {}  # {'[None, None, 2, None, None]': '1_1', '[None, None, 3, None, None]': '10_13',...}
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

        self.transaction = min_trans
        self.state_desc = new_state_info
        self.state_space = list(self.transaction.keys())
        self.start_state = start_state_p
        self.obs_minobs_map = _obs_minobs_map
        self.minobs_obs_map = _minobs_obs_map
        self.minimized = True

        # save minimized_files
        minimized_files = [self.transaction, self.state_desc, self.state_space, self.start_state, self.obs_minobs_map, self.minobs_obs_map, self.minimized]
        with open(os.path.join(self.mmn_directory,'minimized_files.p'), 'wb') as handle:
            pickle.dump(minimized_files, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
    @staticmethod
    def traverse_compatible_states(states, compatibility_mat):
        '''
        Used in minimizing states.
        '''
        for i, s in enumerate(states):
            for j, s_next in enumerate(states[i + 1:]):
                compatible = True
                # input state와 compatibility_mat가 True인 모든 쌍들을 구하고 ([1,0],[3,0],[6,0],...,[18,0])
                # 그 각각 모든 entry끼리 전부 compatibility_mat가 True ( (1,3), (1,6), ... ) 가 아니면 break
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
    
    def evaluate(self, num_episodes, seed):
        with open(os.path.join(self.mmn_directory, 'minimized_files.p'), 'rb') as handle:
            m_files = pickle.load(handle)

        self.transaction = m_files[0]
        self.state_desc = m_files[1]
        self.state_space = m_files[2]
        self.start_state = m_files[3]
        self.obs_minobs_map = m_files[4]
        self.minobs_obs_map = m_files[5]
        self.minimized = m_files[6]

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model.eval()
        total_reward = 0
        # For convenience
        # MM_net = self.model
        # Comm_net = self.model.call_model()

        # with torch.no_grad():

        #     for ep in range(num_episodes):
        #         x = self.env.reset(epoch=0)
        #         prev_state = self.model.policy_net.init_hidden(batch_size=x.shape[0])
        #         ep_reward = 0
        #         # curr_state = self.second_state[np.random.choice(len(self.second_state))]  # random from second_state
        #         # TODO: quantized는 agent_0기준으로 했지만, 갈아끼는건 모든 agent를 갈아낀다.

        #         # TODO: t=0초일 때, prev_hid가 0인데
        #         # transaction table에는 
        #         ipdb.set_trace()
        #         for t in range(self.args.max_steps):
        #             x = Comm_net.encoder(x)
        #             _, q_x = MM_net.obs_qb_net(x)
        #             pass
                    