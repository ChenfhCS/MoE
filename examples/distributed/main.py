import argparse
import time, datetime
import math
import os, sys
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist

import warnings
import pytz

from utils import create_exp_dir
from modeling import Create_MoE_Model, save_model
from training import train_xl_MoE, train_Bert_MoE, train_GPT_MoE

# 忽略所有UserWarning
warnings.simplefilter(action='ignore', category=Warning)

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
# model and training configs
parser.add_argument('--model_name', type=str, choices=['xl', 'bert', 'gpt'],
                    help='model name')
parser.add_argument('--train_batch_size', type=int, default=4,
                    help='train batch size')
parser.add_argument('--eval_batch_size', type=int, default=4,
                    help='eval batch size')
parser.add_argument('--num_epochs', type=int, default=1,
                    help='number of epochs')
# system configs
parser.add_argument('--use_wandb', action='store_true',
                    help='whether use wandb for logging')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--log_interval', type=int, default=10,
                    help='report interval')
parser.add_argument('--eval_interval', type=int, default=10,
                    help='report interval')
parser.add_argument('--work_dir', default='LM-TFM', type=str,
                    help='experiment directory.')
# moe configs
parser.add_argument('--moe', action='store_true',
                    help='replace position-wise ffn with moe position-wise ffn')
parser.add_argument('--moe-num-experts', type=int, default=64,
                    help='number of experts in MoE')
parser.add_argument('--moe-top-k', type=int, default=2,
                    help='top_k experts in hard gate of moe')
parser.add_argument('--fuse_token', action='store_true',
                    help='whether to fuse tokens')
parser.add_argument('--expert_parallel', action='store_true',
                    help='expert parallel')
parser.add_argument('--moe_world_size', type=int, default=1,
                    help='number of devices for expert parallelism')
args = parser.parse_args()
assert args.moe_num_experts >= args.moe_top_k, "must have moe-num-expert >= moe-top_k"

# set environment
device = torch.device('cuda' if args.cuda else 'cpu')
if args.expert_parallel:
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
else:
    local_rank = 0
    global_rank = 0
    world_size = 1

# log file
tokyo = pytz.timezone('Asia/Tokyo')
# 获取当前东京时间
current_tokyo_time = datetime.datetime.now(tokyo)
# 格式化时间
time_stamp = current_tokyo_time.strftime('%Y%m%d-%H%M%S')
args.work_dir = '{}'.format(args.work_dir)
log_suffix = time_stamp + f'[{str(args.moe_num_experts)}Exp_Fusion_{args.fuse_token}_top{args.moe_top_k}]'
args.work_dir = os.path.join(args.work_dir, log_suffix)
if local_rank == 0:
    logging = create_exp_dir(args.work_dir, debug=args.debug)
else:
    logging = None

# ep: expert parallel; dp: data parallel
ep_group_world_size = args.moe_world_size #Within a group, all GPUs use expert parallel
ep_group_rank = global_rank // ep_group_world_size # which expert parallel group the GPU belongs to 
dp_group_world_size = world_size // ep_group_world_size # how many groups
dp__group_rank = global_rank % ep_group_world_size # which data parallel group the GPU belongs to

# initialize dist group
if args.expert_parallel:
    dist.init_process_group(backend='nccl',
                            # init_method='tcp://127.0.0.1:8000',
                            init_method='env://',
                            world_size=world_size,
                            rank=global_rank)
    # expert parallel group
    for j in range(dp_group_world_size):
        moe_comm_group_list = [i + ep_group_world_size * j for i in range(ep_group_world_size)]
        group = torch.distributed.new_group(moe_comm_group_list)
        if j == ep_group_rank:
            moe_comm_group = group
            print("rank {}/{}, moe_comm_group list is {}".format(global_rank+1, world_size, moe_comm_group_list))
    # data parallel group
    for j in range(ep_group_world_size):
        moe_sync_group_list = [j + ep_group_world_size * i for i in range(dp_group_world_size)]  # GPUs use the same experts (from different group) will share parameters
        group = torch.distributed.new_group(moe_sync_group_list)
        if j == dp__group_rank:
            moe_sync_group = group
            print("rank {}/{}, moe_sync_group list is {}".format(global_rank, world_size, moe_sync_group_list))
else:
    moe_comm_group = None
    moe_sync_group = None

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed_all(args.seed)

# load model
model, tokenizer = Create_MoE_Model(model_name = args.model_name, 
                                    moe = args.moe, moe_num_experts = args.moe_num_experts // ep_group_world_size,
                                    moe_top_k = args.moe_top_k, moe_group = moe_comm_group,
                                    moe_world_size = ep_group_world_size)
if args.model_name == 'xl':
    args.n_layer = model.config.n_layer
    args.n_embd = model.config.d_embed
    args.n_inner = model.config.d_inner
elif args.model_name == 'gpt': # xl and gpt
    args.n_layer = model.config.n_layer
    args.n_embd = model.config.n_embd
    args.n_inner = model.config.n_inner
elif args.model_name == 'bert': # bert
    args.n_layer = model.config.num_hidden_layers
    args.n_embd = model.config.hidden_size
    args.n_inner = model.config.intermediate_size
else:
    raise AttributeError("No such models!")
args.n_all_param = sum([p.nelement() for p in model.parameters()])

if local_rank == 0:
    logging('=' * 100)
    for k, v in args.__dict__.items():
        logging('    - {} : {}'.format(k, v))
    logging('=' * 100)


model = model.to(device)

# training
if args.model_name == 'xl':
    train_xl_MoE(device = device, dist = args.expert_parallel,
                 model = model, tokenizer= tokenizer,
                 train_batch_size = args.train_batch_size,
                 eval_batch_size = args.eval_batch_size,
                 world_size = world_size,
                 global_rank = global_rank,
                 local_rank = local_rank,
                 moe_sync_group = moe_sync_group,
                 num_epochs = args.num_epochs, logger = logging,
                 log_interval = args.log_interval,
                 eval_interval = args.eval_interval,
                 use_wandb = args.use_wandb)
elif args.model_name == 'bert':
    train_Bert_MoE(device = device, dist = args.expert_parallel,
                 model = model, tokenizer= tokenizer,
                 train_batch_size = args.train_batch_size,
                 eval_batch_size = args.eval_batch_size,
                 world_size = world_size,
                 global_rank = global_rank,
                 local_rank = local_rank,
                 moe_sync_group = moe_sync_group,
                 num_epochs = args.num_epochs, logger = logging,
                 log_interval = args.log_interval,
                 eval_interval = args.eval_interval,
                 use_wandb = args.use_wandb)
elif args.model_name == 'gpt':
    train_GPT_MoE(device = device, dist = args.expert_parallel,
                 model = model, tokenizer= tokenizer,
                 train_batch_size = args.train_batch_size,
                 eval_batch_size = args.eval_batch_size,
                 world_size = world_size,
                 global_rank = global_rank,
                 local_rank = local_rank,
                 moe_sync_group = moe_sync_group,
                 num_epochs = args.num_epochs, logger = logging,
                 log_interval = args.log_interval,
                 eval_interval = args.eval_interval,
                 use_wandb = args.use_wandb)
else:
    raise Exception('Error: no such a model named {}'.format(args.model_name))
