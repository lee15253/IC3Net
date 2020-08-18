import sys
import time
import signal
import argparse
import os

import numpy as np
import torch
import visdom
import data
from models import *
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args
from storage import Storage
from trainer import Trainer
from qbn_trainer import QBNTrainer
from torch.utils.tensorboard import SummaryWriter

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch RL trainer')
# training
# note: number of steps = num_rollout_steps x n_agents
parser.add_argument('--dest', type=str, default='',
                    help='destination for the training logs to be saved')
parser.add_argument('--num_train_rollout_steps', type=int, default=30000,
                    help='number of steps to collect trajectory for training')
parser.add_argument('--num_test_rollout_steps', type=int, default=5000,
                     help='number of steps to collect trajectory for testing')
parser.add_argument('--storage_size', type=int, default=100000,
                    help='size of storage to store the trajectory')
parser.add_argument('--noisy_rollouts', action='store_true', default=False,
                    help='perform noisy rollouts to get data diversity in training')
parser.add_argument('--batch_size', type=int, default=128,
                    help='number of batch size to train the quanitzed bottleneck network')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train the quantized bottleneck network')

# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')

# QBN
parser.add_argument('--qbn', action='store_true', default=False,
                    help='use quantized bottle-neck architecture')
parser.add_argument('--obs_quantize_size', default=32, type=int,
                    help='hidden bottle neck size for observation')
parser.add_argument('--comm_quantize_size', default=32, type=int,
                    help='hidden bottle neck size for communication')
parser.add_argument('--hidden_quantize_size', default=32, type=int,
                    help='hidden bottle neck size for hidden-states')

# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed') # TODO: works in thread?
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
# environment
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
# other
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--save', default='', type=str,
                    help='save the model after training')
parser.add_argument('--save_every', default=0, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')
parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")

# CommNet specific args
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--nagents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=1,
                    help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="Whether communication should be there")
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                    + ' Default 10000 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true',
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')


init_args_for_env(parser)
args = parser.parse_args()

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0

    # For TJ set comm action to 1 as specified in paper to showcase
    # importance of individual rewards even in cooperative games
    if args.env_name == "traffic_junction":
        args.comm_action_one = True
# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

env = data.init(args.env_name, args, False)

num_inputs = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)): # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'


parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)

print(args)


if args.commnet:
    policy_net = CommNetMLP(args, num_inputs)
elif args.random:
    policy_net = Random(args, num_inputs)
elif args.recurrent:
    policy_net = RNN(args, num_inputs)
else:
    policy_net = MLP(args, num_inputs)

if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

# num-process is always 1
trainer = Trainer(args, policy_net, data.init(args.env_name, args))

disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
disp_trainer.display = True
def disp():
    x = disp_trainer.get_episode()

log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')

if args.plot:
    vis = visdom.Visdom(env=args.plot_env)

def run():
    begin_time = time.time()
    stat = dict()

    # 1. Initialize QBN model
    print('Initialize QBN Model')
    obs_qb_net = HxQBNet(input_size=args.hid_size, x_features=args.obs_quantize_size)
    comm_qb_net = HxQBNet(input_size=args.hid_size, x_features=args.comm_quantize_size)
    hidden_qb_net = HxQBNet(input_size=args.hid_size, x_features=args.hidden_quantize_size)

    # 2. Initialize QBN trainer
    print('Initialize QBN Trainer')
    storage = Storage(storage_size=args.storage_size,
                      n_agents=args.nagents,
                      hid_size=args.hid_size,
                      num_actions=args.num_actions[0])
    writer = SummaryWriter(os.path.dirname(args.load) + '/' + args.dest)
    env = data.init(args.env_name, args)
    qbn_trainer = QBNTrainer(args, env, policy_net, obs_qb_net, comm_qb_net, hidden_qb_net, storage, writer)

    # 3. Collect Trajectory from the trained model
    print('Collect trajectory from the trained model')
    qbn_trainer.perform_rollouts(args.num_train_rollout_steps, store=True)

    # 4. Train & Insert QBN (check the performance iteratively)
    print('Train QBN model')
    qbn_trainer.train_all()
    qbn_trainer.test_all()

    # 5. Fine-tune Network


    # 6. Minimization or Functional Pruning




def save(path):
    d = dict()
    dir = os.path.dirname(path)
    if not (os.path.exists(dir)):
        os.makedirs(dir)
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    torch.save(d, path)

def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])

def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if args.load != '':
    load(args.load)

run()
if args.display:
    env.end_display()

if args.save != '':
    save(args.save)

if sys.flags.interactive == 0 and args.nprocesses > 1:
    trainer.quit()
    import os
    os._exit(0)
