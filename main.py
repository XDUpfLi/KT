import os
import json
import torch
import logging
import argparse
import numpy as np
from torch import nn
from res10 import res10
from models import conv64
from res12 import resnet12
from core import EvaluateFewShot
from datasets import MiniImageNet
from types import SimpleNamespace
from cdfsl import make_cdfsl_loader
from collections import OrderedDict
from hebb import kt
from utils import prepare_meta_batch, make_task_loader


class ModelWrapper(nn.Module):
  def __init__(self, embed, fc_sizes):
    super(ModelWrapper, self).__init__()
    self.embed = embed
    self.feature_index = [-1]
    seq = []

    for i in range(len(fc_sizes)-2):
      seq += [nn.Linear(fc_sizes[i], fc_sizes[i+1]), nn.ReLU(), nn.Dropout(0.5)]

    seq += [nn.Linear(fc_sizes[-2], fc_sizes[-1])]
    self.output_layer = nn.Sequential(*seq)

  def forward(self, x, output_layer=True, return_feature=False, addse=False):
    if addse:
        self.x = [self.embed(x, return_feature=True)]
        for m in self.output_layer:
            self.x += [m(self.x[-1][0].flatten(1))]
        if return_feature:
            return self.x[-1], self.x[-2][0], self.x[-2][1]
        else:
            return self.x[-1]
    else:
        self.x = [self.embed(x)]
        for m in self.output_layer:
          self.x += [m(self.x[-1].flatten(1))]
        if return_feature:
            return self.x[-1], self.x[-2]
        else:
            return self.x[-1]

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('config', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--n', type=int)
parser.add_argument('--k', type=int)
parser.add_argument('--q', type=int)
parser.add_argument('--eval-batches', type=int)
parser.add_argument('--gpu', type=int, nargs='+')
parser.add_argument('--num-workers', type=int)
parser.add_argument('--model', type=str)
parser.add_argument('--dropout', type=float)
parser.add_argument('--sparserate', type=float)
parser.add_argument('--balance', type=float)
parser.add_argument('--hebb-lr', type=float)
parser.add_argument('--inner-val-steps', type=int)
parser.add_argument('--meta-batch-size', type=int)
parser.add_argument('--backbone', choices=['conv64', 'res12', 'res10', 'res18'])
parser.add_argument('--seed', type=int)
parser.add_argument('--feature-index', type=int, nargs='+')
args = parser.parse_args()
with open(args.config) as f:
  config = json.load(f)
# override config with cmd line args
config.update(vars(args))
args = SimpleNamespace(**config)

exp_name = args.model.split('/')
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler('{}/{}s_b{}_{}shot_{}_{}.txt'.format(exp_name[0], args.sparserate, args.balance, args.n, args.dataset, args.backbone))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

assert(torch.cuda.is_available())
device = torch.device(args.gpu[0])

eval_few_shot_args = {
  'num_tasks': args.eval_batches,
  'n_shot': args.n,
  'k_way': args.k,
  'q_queries': args.q,
  'prepare_batch': prepare_meta_batch(
      args.n, args.k, args.q, args.meta_batch_size, 2, device),
  'inner_train_steps': args.inner_val_steps,
  'hebb_lr': args.hebb_lr,
  'device': device,
  'sparserate': args.sparserate,
  'balance': args.balance,
  'xdom': hasattr(args, 'dataset'),
}

state_dict = torch.load(args.model, map_location={'cuda:1': 'cuda:0'})

if args.backbone == 'res12':
  embed = resnet12(avg_pool=True, drop_rate=args.dropout, dropblock_size=5)
  fc_sizes = [640, 80]
elif args.backbone == 'res10':
  embed = res10()
  fc_sizes = [512, 80]
elif args.backbone == 'conv64':
  embed = conv64()
  fc_sizes = [64, 80]

model = ModelWrapper(embed, fc_sizes)
model.feature_index = args.feature_index
model = nn.DataParallel(model, device_ids=args.gpu)
model.load_state_dict(state_dict)
model = model.to(device, dtype=torch.double)
model.eval()

if hasattr(args, 'dataset'):
  test_loader = make_cdfsl_loader(args.dataset,
                                  args.eval_batches,
                                  args.n,
                                  args.k,
                                  args.q,
                                  small=(args.backbone!='res10'))
else:
  test_loader = make_task_loader(MiniImageNet('test',
                                              small=(args.backbone!='res10')),
                                 args, train=False, meta=True)

loss_fn = nn.CrossEntropyLoss().cuda()

evaluator = EvaluateFewShot(eval_fn=kt,
                            taskloader=test_loader,
                            **eval_few_shot_args)

logs = {
  'dataset': args.dataset if hasattr(args, 'dataset') else 'miniImagenet',
  'n-shot': args.n,
  'feature_index': args.feature_index,
  'hebb_lr': args.hebb_lr,
  'inner_val_steps': args.inner_val_steps,
  'sparserate': args.sparserate,
  'balance': args.balance,
}
evaluator.model = {'sys1': model}
evaluator.optimiser = None
evaluator.loss_fn = loss_fn
evaluator.on_epoch_end(0, logs)
logger.info(logs)
