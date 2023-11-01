import argparse
import tqdm, os, pytz, time, math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from gpt2.data import Dataset
from gpt2.training import TrainingSpec, TrainConfig, Recorder
from gpt2.utils import fusing
from gpt2.modeling import Transformer
from gpt2.data import Dataset, Vocab, TokenizedCorpus
from gpt2.training import TrainConfig, TrainingSpec, Trainer
from typing import Tuple, Iterator, Dict, Optional

from utils.exp_utils import create_exp_dir
from fmoe.distributed import DistributedGroupedDataParallel as DDP

try:
    from apex import amp
except ModuleNotFoundError:
    pass

import warnings
warnings.filterwarnings(action='ignore')

class GPT2TrainingSpec(TrainingSpec):
    def __init__(self, train_corpus: str, eval_corpus: str, vocab_path: str,
                 seq_len: int, layers: int, heads: int, dims: int, rate: int,
                 dropout: float, base_lr: float, wd_rate: float,
                 total_steps: int, use_grad_ckpt: bool,
                 moe: bool, moe_num_expert: int, 
                 moe_world_size: int, moe_group,
                 moe_top_k: int, fuse_token: bool):
        self.train_corpus = train_corpus
        self.eval_corpus = eval_corpus
        self.vocab_path = vocab_path
        self.seq_len = seq_len
        self.layers = layers
        self.heads = heads
        self.dims = dims
        self.rate = rate
        self.dropout = dropout
        self.base_lr = base_lr
        self.wd_rate = wd_rate
        self.total_steps = total_steps
        self.use_grad_ckpt = use_grad_ckpt

        # moe configs
        self.moe = moe
        self.moe_num_expert = moe_num_expert
        self.moe_world_size = moe_world_size
        self.moe_group = moe_group
        self.moe_top_k = moe_top_k
        self.fuse_token = fuse_token

    def initialize(self):
        self.vocab = Vocab(vocab_path=self.vocab_path)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_idx,
                                             reduction='mean')

    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        train_dataset = TokenizedCorpus(corpus_path=self.train_corpus,
                                        vocab=self.vocab,
                                        seq_len=self.seq_len)
        eval_dataset = TokenizedCorpus(corpus_path=self.eval_corpus,
                                       vocab=self.vocab,
                                       seq_len=self.seq_len)
        return train_dataset, eval_dataset

    def construct_model(self) -> nn.Module:
        return Transformer(layers=self.layers, pad_idx=self.vocab.pad_idx,
                           words=len(self.vocab), seq_len=self.seq_len,
                           heads=self.heads, dims=self.dims, 
                           rate=self.rate, dropout=self.dropout, bidirectional=False,
                           moe=self.moe, moe_num_expert=self.moe_num_expert,
                           moe_world_size=self.moe_world_size, moe_group=self.moe_group,
                           moe_top_k=self.moe_top_k, fuse_token=self.fuse_token)

    def create_optimizer(self, params: Iterator[nn.Parameter]
                         ) -> Tuple[optim.Optimizer,
                                    optim.lr_scheduler._LRScheduler]:
        optimizer = fusing.Adam(
            params, lr=self.base_lr, weight_decay=self.wd_rate)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1 - step / self.total_steps)
        return optimizer, scheduler

    def train_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                        ) -> Dict[str, torch.Tensor]:
        if self.moe is False:
            logits = model(data['input'], use_grad_ckpt=self.use_grad_ckpt)
        else:
            logits, fusion_costs = model(data['input'], use_grad_ckpt=self.use_grad_ckpt)
        loss = self.criterion(logits.transpose(1, 2), data['output'])

        if self.moe is False:
            return {'loss': loss}
        else:
            return {'loss': loss, 'fusion_costs':fusion_costs}

    def eval_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                       ) -> Dict[str, torch.Tensor]:
        logits, _ = model(data['input'], past=None)
        loss = self.criterion(logits.transpose(1, 2), data['output'])
        return {'loss': loss}

def _fetch_from(args, world_size, dataset: Dataset, rank: int, batch: int
                    ) -> Dict[str, torch.Tensor]:
        if args.multi_gpu:
            # In distributed training environment, each process must ignore
            # sub-batches of other processes and fetch corresponding one only.
            batch = batch // world_size

            dataset.skip(rank * batch)
            data = dataset.fetch(batch)
            dataset.skip((world_size - rank - 1) * batch)
        else:
            data = dataset.fetch(batch)

        return {k: v.cuda() for k, v in data.items()}

# configs
parser = argparse.ArgumentParser(prog='gpt2', description='PyTorch implementation of OpenAI GPT-2')
# normal configs
parser.add_argument('--train_corpus', required=True,
                       help='training corpus file path')
parser.add_argument('--eval_corpus', required=True,
                       help='evaluation corpus file path')
parser.add_argument('--vocab_path', required=True,
                    help='vocabulary file path')
parser.add_argument('--work_dir', default='works', type=str,
                    help='experiment directory.')
parser.add_argument('--seq_len', default=64, type=int,
                    help='maximum sequence length')
parser.add_argument('--layers', default=12, type=int,
                    help='number of transformer layers')
parser.add_argument('--heads', default=16, type=int,
                    help='number of multi-heads in attention layer')
parser.add_argument('--dims', default=1024, type=int,
                    help='dimension of representation in each layer')
parser.add_argument('--rate', default=4, type=int,
                    help='increase rate of dimensionality in bottleneck')
parser.add_argument('--dropout', default=0.1, type=float,
                    help='probability that each element is dropped')
parser.add_argument('--batch_train', default=64, type=int,
                    help='number of training batch size')
parser.add_argument('--batch_eval', default=64, type=int,
                    help='number of evaluation batch size')
parser.add_argument('--base_lr', default=1e-4, type=float,
                    help='default learning rate')
parser.add_argument('--wd_rate', default=1e-2, type=float,
                    help='weight decay rate')
parser.add_argument('--total_steps', default=1000000, type=int,
                    help='number of total training steps')
parser.add_argument('--eval_steps', default=500, type=int,
                    help='period to evaluate model and record metrics')
parser.add_argument('--save_steps', default=1000, type=int,
                    help='period to save training state to checkpoint')
parser.add_argument('--save_model_path', default='model.pth',
                    help='save trained model weights to the file')
parser.add_argument('--save_checkpoint_path', default='checkpoint.pth',
                    help='save training state to the checkpoint file')
parser.add_argument('--from_checkpoint', default=None,
                    help='load last training state from checkpoint file')
parser.add_argument('--from_pretrained', default=None,
                    help='initialize parameters from pretrained model')
parser.add_argument('--use_amp', action='store_true',
                    help='use automatic mixed-precision in training')
parser.add_argument('--use_grad_ckpt', action='store_true',
                    help='use gradient checkpointing in transformer layers')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--log-interval', type=int, default=100,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=3000,
                    help='evaluation interval')
# moe configs
parser.add_argument('--moe', action='store_true',
                    help='replace position-wise ffn with moe position-wise ffn')
parser.add_argument('--moe-num-expert', type=int, default=64,
                    help='number of experts in MoE')
parser.add_argument('--moe-top-k', type=int, default=2,
                    help='top_k experts in hard gate of moe')
parser.add_argument('--expert_parallel', action='store_true',
                    help='expert parallel')
parser.add_argument('--fuse_token', action='store_true',
                    help='whether to fuse tokens')
args = parser.parse_args()

assert args.moe_num_expert >= args.moe_top_k, "must have moe-num-expert >= moe-top_k"

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

group_world_size = world_size #Within a group, all GPUs use expert parallel
group_rank = global_rank // group_world_size # which expert parallel group the GPU belongs to 
group_size = world_size // group_world_size # how many groups
inner_group_rank = global_rank % group_world_size # which data parallel group the GPU belongs to

if args.multi_gpu:
    if args.expert_parallel:
        dist.init_process_group(backend='nccl',
                                # init_method='tcp://127.0.0.1:8000',
                                init_method='env://',
                                world_size=world_size,
                                rank=global_rank)
        # expert parallel group
        for j in range(group_size):
            moe_comm_group_list = [i + group_world_size * j for i in range(group_world_size)]
            group = torch.distributed.new_group(moe_comm_group_list)
            if j == group_rank:
                moe_comm_group = group
                print("rank {}/{}, moe_comm_group list is {}".format(global_rank+1, world_size, moe_comm_group_list))
else:
    moe_comm_group = None

# log file
# 设置东京时区
tokyo = pytz.timezone('Asia/Tokyo')
# 获取当前东京时间
current_tokyo_time = datetime.now(tokyo)
# 格式化时间
time_stamp = current_tokyo_time.strftime('%Y%m%d-%H%M%S')
args.work_dir = '{}'.format(args.work_dir)
log_suffix = time_stamp + f'[{str(args.moe_num_expert)}Exp_Fusion_{args.fuse_token}_top{args.moe_top_k}]'
args.work_dir = os.path.join(args.work_dir, log_suffix)
if local_rank == 0:
    logging = create_exp_dir(args.work_dir, debug=args.debug)

# Initialize training environment and prepare datasets.
spec = GPT2TrainingSpec(
        train_corpus=args.train_corpus, eval_corpus=args.eval_corpus,
        vocab_path=args.vocab_path, seq_len=args.seq_len, layers=args.layers,
        heads=args.heads, dims=args.dims, rate=args.rate, dropout=args.dropout,
        base_lr=args.base_lr, wd_rate=args.wd_rate,
        total_steps=args.total_steps, use_grad_ckpt=args.use_grad_ckpt,
        moe=args.moe, moe_num_expert=args.moe_num_expert // world_size, 
        moe_world_size=world_size, moe_group=moe_comm_group,
        moe_top_k=args.moe_top_k, fuse_token=args.fuse_token)
spec.initialize()

# Load data
train_dataset, eval_dataset = spec.prepare_datasets()

# Build the model
model = spec.construct_model().cuda()
args.n_all_param = sum([p.nelement() for p in model.parameters()])
if local_rank == 0:
    logging('=' * 100)
    for k, v in args.__dict__.items():
        logging('    - {} : {}'.format(k, v))
    logging('=' * 100)
    logging('#params = {}'.format(args.n_all_param))
if args.from_pretrained:
    ckpt = torch.load(args.from_pretrained, map_location='cuda')
    model.load_state_dict(ckpt['model'])

    # Because the weights data allocates quite a lot of GPU memories,
    # we need to free the memories explicitly.
    del ckpt
    torch.cuda.empty_cache()

# Create an optimizer and learning rate scheduler.
optimizer, scheduler = spec.create_optimizer(model.parameters())
recorder = Recorder()

if args.use_amp:
    model, optimizer = amp.initialize(
        model, optimizer, opt_level='O2', verbosity=0)

# distributed settings
if args.multi_gpu:
    if args.expert_parallel:
        model.cuda(local_rank)
        # data parallel group
        for j in range(group_world_size):
            moe_sync_group_list = [j + group_size * i for i in range(group_size)]  # GPUs use the same experts (from different group) will share parameters
            group = torch.distributed.new_group(moe_sync_group_list)
            if j == inner_group_rank:
                moe_sync_group = group
                print("rank {}/{}, moe_sync_group list is {}".format(global_rank, world_size, moe_sync_group_list))
        model = DDP(model, device_ids=[local_rank], moe_sync_group = moe_sync_group)
        model._sync_params()
        # model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        # model.cuda(local_rank)
    else:
        model = nn.DataParallel(model, dim=1).to(device)
else:
    model = model.to(device)

start_step = 0
# Restore last training states from checkpoint.
if args.from_checkpoint:
    ckpt = torch.load(args.from_checkpoint, map_location='cuda')

    start_step = ckpt['step']
    recorder = ckpt['recorder']

    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])

    train_dataset.assign(ckpt['train_dataset'])
    eval_dataset.assign(ckpt['eval_dataset'])

    if args.use_amp:
        amp.load_state_dict(ckpt['amp'])

    # Because the checkpoint data allocates quite a lot of GPU
    # memories, we need to free the memories explicitly.
    del ckpt
    torch.cuda.empty_cache()

# training
def train():
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    loss_log = []
    model.train()
    training_iters = range(start_step + 1, args.total_steps)
    for step in training_iters:
        total_fusion_costs = 0
        model.zero_grad()
        data = _fetch_from(args, world_size, train_dataset, local_rank, args.batch_train)
        metrics = spec.train_objective(data, model)
        loss = metrics['loss']
        train_loss += loss.float().item()

        if args.moe is True:
            total_fusion_costs += metrics['fusion_costs']
        if args.use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # expert parallel
        if args.expert_parallel:
            model.allreduce_params()
        optimizer.step()
        scheduler.step()

        train_step += 1
        if train_step % args.log_interval == 0:
            if local_rank == 0:
                cur_loss = train_loss / args.log_interval
                elapsed = time.time() - log_start_time
                log_str = '| step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                        '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                    train_step, step, optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss)
                log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
                if args.moe is True:
                    log_str += ' | fusion costs {:5.2f}'.format(total_fusion_costs*1000)
                loss_log.append(round(cur_loss, 2))
                if len(loss_log) % 10 == 0:
                    log_str += ' | current losses {}'.format(loss_log)
                logging(log_str)
                log_start_time = time.time()
            train_loss = 0
            total_fusion_costs = 0

        if train_step == args.total_steps:
            break


# Loop over epochs.
train_step = 0
train_loss = 0
best_val_loss = None

log_start_time = time.time()
eval_start_time = time.time()

# At any point you can hit Ctrl + C to break out of training early.
try:
    train()
    if train_step == args.total_steps:
        if local_rank == 0:
            logging('-' * 100)
            logging('End of training')
except KeyboardInterrupt:
    if local_rank == 0:
        logging('-' * 100)
        logging('Exiting from training early')

# if local_rank == 0:
#     # Load the best saved model.
#     with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
#         model = torch.load(f)
#     model = model.to(device)

    # # Run on test data.
    # test_loss = evaluate(te_iter)
    # logging('=' * 100)
    # if args.dataset in ['enwik8', 'text8']:
    #     logging('| End of training | test loss {:5.2f} | test bpc {:9.5f}'.format(
    #         test_loss, test_loss / math.log(2)))
    # else:
    #     logging('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(
    #         test_loss, math.exp(test_loss)))
    # logging('=' * 100)

