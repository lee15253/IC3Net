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
                os=np.array([]), start_state=0, total_actions=None):
        
        self.args = args
        self.transaction = t
        self.state_desc = sd
        self.state_space = ss
        self.obs_space = os
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

        q_parameter = torch.load(mmn_directory)  # fine-tuned (final) parameter
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
                                           net_type = 'fine_tuned_model', store = True)
        q_x_batch, q_c_batch, q_h_batch = self.qbn_trainer.storage.fetch_fsm_data()                                  
        ipdb.set_trace()
        self.update_fsm()

    def update_fsm(self):
        """
        udpate self.transaction, self.state_desc
        """
        # TODO: 필요한것: q_x_t, q_c_t, q_h_t, q_h_t+1, a_t, a_t+1 