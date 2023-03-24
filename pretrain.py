import os
import json
import torch
import logging
import argparse
import datetime
from torch import nn
from res10 import res10
from models import conv64
from res12 import resnet12
from types import SimpleNamespace
from utils import adjust_learning_rate
from torch.utils.data import DataLoader
from datasets import MiniImagenetHorizontal


class ModelWrapper(nn.Module):
  def __init__(self, embed, fc_sizes):
    super(ModelWrapper, self).__init__()
    self.embed = embed
    seq = []
    for i in range(len(fc_sizes)-2):
      seq += [nn.Linear(fc_sizes[i], fc_sizes[i+1]), nn.ReLU(), nn.Dropout(0.5)]
    seq += [nn.Linear(fc_sizes[-2], fc_sizes[-1])]
    self.output_layer = nn.Sequential(*seq)
  
  def forward(self, x):
    x = self.embed(x)
    return self.output_layer(x)

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('config', type=str)
# parser.add_argument('--dataset', type=str)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--momentum', type=float)
parser.add_argument('--weight-decay', type=float)
parser.add_argument('--epochs', type=int)
parser.add_argument('--gpu', type=int, nargs='+')
parser.add_argument('--seed', type=int)
parser.add_argument('--dropout', type=float)
parser.add_argument('--backbone', choices=['conv64', 'res12', 'res10'])
parser.add_argument('--num-workers', type=int)
# parser.add_argument('--relu-out', action='store_true')
# parser.add_argument('--flag', type=bool)
args = parser.parse_args()
with open(args.config) as f:
  config = json.load(f)
# override config with cmd line args
config.update(vars(args))
args = SimpleNamespace(**config)

assert(torch.cuda.is_available())
device = torch.device(args.gpu[0])

train_data = MiniImagenetHorizontal('train', small=(args.backbone!='res10'))
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
                          num_workers=args.num_workers, drop_last=True)
val_data = MiniImagenetHorizontal('val', small=(args.backbone!='res10'))
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, 
                        num_workers=args.num_workers, drop_last=False)

runid = datetime.datetime.now().strftime('%y%m%dT%H%M%S') + 'P{}'.format(os.getpid())
exp_name = args.config.split('/')[1].split('.')[0]

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler('./model_result_log/{}_{}_{}.txt'.format(exp_name, runid, args.backbone))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

if args.backbone == 'res12':
  embed = resnet12(avg_pool=True, drop_rate=args.dropout, dropblock_size=5)
  fc_sizes = [640, len(train_data.classes)]
elif args.backbone == 'res10':
  embed = res10()
  fc_sizes = [512, len(train_data.classes)]
elif args.backbone == 'conv64':
  embed = conv64()
  fc_sizes = [64, len(train_data.classes)]
model = ModelWrapper(embed, fc_sizes)

model = model.to(device)
model = nn.DataParallel(model, device_ids=args.gpu)
lossf = nn.CrossEntropyLoss().to(device)
optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

best_acc = 0.0
for epoch in range(args.epochs):
  adjust_learning_rate(optim, epoch, args.lr, args.epochs)
  model.train()
  for i, (x, y) in enumerate(train_loader):
    x, y = x.to(device), y.to(device)
    y_hat = model(x)
    loss = lossf(y_hat, y)
    acc = (y_hat.argmax(dim=1) == y).float().mean()
    logger.info('epoch: {}/{} iter: {}/{} loss: {} acc: {}%'.format(epoch, args.epochs, i, len(train_loader), loss.item(), acc.item()*100))
    optim.zero_grad()
    loss.backward()
    optim.step()
    pass

  model.eval()
  acc = []
  with torch.no_grad():
    for x, y in val_loader:
      x, y = x.to(device), y.to(device)
      y_hat = model(x)
      acc.append((y_hat.argmax(dim=1) == y).float().mean())

  acc  = sum(acc) / len(acc)
  if acc > best_acc:
    torch.save(model.state_dict(), './model_result_log_first_submit/{}_{}_{}_maxacc.pth'.format(exp_name, runid, args.backbone))
    best_acc = acc
