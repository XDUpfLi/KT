import torch
import json
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict
from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict, List, Callable, Union
from core import create_nshot_task_label, EvaluateFewShot
from metrics import categorical_accuracy
from utils import correlation

def rand_prop(imgs, training, sparserate=0.5):
    b, c, w, h = imgs.shape
    drop_rate = sparserate
    drop_rates = torch.FloatTensor(np.ones((w, h)) * drop_rate)
    if training:
        masks = torch.bernoulli(1. - drop_rates).type(dtype=torch.float64)
        imgs = masks.cuda() * imgs
    else:
        imgs = imgs * (1. - drop_rate)
    return imgs


def kt(model: Module,
              optimiser: Optimizer,
              loss_fn: Callable,
              x: torch.Tensor,
              y: torch.Tensor,
              n_shot: int,
              k_way: int,
              q_queries: int,
              inner_train_steps: int,
              hebb_lr: float,
              train: bool,
              sparserate: float,
              balance: float,
              device: Union[str, torch.device],
              xdom=False):

    args = {'dtype': torch.double, 'requires_grad': True}
    task_predictions = []
    task_losses = []

    for x_ in x:
        if xdom:
            x_ = x_.reshape(k_way, n_shot + q_queries, *x_.shape[1:])
            x_support = x_[:, :n_shot].flatten(0, 1)
            x_query = x_[:, n_shot:].flatten(0, 1)
        else:
            x_support = x_[:n_shot * k_way]
            x_query = x_[n_shot * k_way:]

        # get feature
        with torch.no_grad():
            _, x_query_feature_1, x_query_feature_4_1 = model(x_query, output_layer=False, return_feature=True, addse=True)
        x_query_feature_1 = x_query_feature_1.t()
        x_query_feature_1 = F.normalize(x_query_feature_1, dim=0)

        x_support_1_1 = rand_prop(x_support, True, sparserate)
        x_support_1_2 = rand_prop(x_support, True, sparserate)
        _, x_support_feature_1_1, x_support_feature_4_1_1 = [], [], []
        with torch.no_grad():
            for i in range(int((n_shot * k_way) / 5)):
                __, x_support_feature_, x_support_feature_4_ = model(x_support_1_1[i * 5:i * 5 + 5], output_layer=False,
                                                                     return_feature=True,
                                                                     addse=True)
                _.append(__)
                x_support_feature_1_1.append(x_support_feature_)
                x_support_feature_4_1_1.append(x_support_feature_4_)
            _, x_support_feature_1_1, x_support_feature_4_1_1 = torch.cat(_), torch.cat(
                x_support_feature_1_1), torch.cat(x_support_feature_4_1_1)

        _, x_support_feature_1_2, x_support_feature_4_1_2 = [], [], []
        with torch.no_grad():
            for i in range(int((n_shot * k_way) / 5)):
                __, x_support_feature_, x_support_feature_4_ = model(x_support_1_2[i * 5:i * 5 + 5], output_layer=False,
                                                                     return_feature=True,
                                                                     addse=True)
                _.append(__)
                x_support_feature_1_2.append(x_support_feature_)
                x_support_feature_4_1_2.append(x_support_feature_4_)
            _, x_support_feature_1_2, x_support_feature_4_1_2 = torch.cat(_), torch.cat(
                x_support_feature_1_2), torch.cat(x_support_feature_4_1_2)

        # se
        w1 = torch.zeros(x_query_feature_1.shape[0] // 4, x_query_feature_1.shape[0], **args).cuda()
        w2 = torch.zeros(x_query_feature_1.shape[0], x_query_feature_1.shape[0] // 4, **args).cuda()
        init.kaiming_normal_(w1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(w2, mode='fan_out', nonlinearity='relu')
        # trans
        p = 1.0e-4 * torch.ones(x_query_feature_1.shape[0], x_query_feature_1.shape[0], **args).cuda()
        # init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
        # cls
        w = 1.0e-4 * torch.ones(k_way*n_shot, k_way*q_queries, **args).cuda()
        for i in range(inner_train_steps):
            fc1_1 = F.linear(x_support_feature_1_1, w1)
            fc1_1 = F.relu(fc1_1)
            fc2_1 = F.linear(fc1_1, w2)
            fc2_1 = F.sigmoid(fc2_1).view(x_support_feature_1_1.shape[0], x_support_feature_1_1.shape[1], 1, 1)
            x_support_feature_se_1_1 = x_support_feature_4_1_1 * fc2_1
            x_support_feature_se_1_1 = F.adaptive_avg_pool2d(x_support_feature_se_1_1, 1).view(x_support_feature_1_1.shape[0], x_support_feature_1_1.shape[1])

            fc1_2 = F.linear(x_support_feature_1_2, w1)
            fc1_2 = F.relu(fc1_2)
            fc2_2 = F.linear(fc1_2, w2)
            fc2_2 = F.sigmoid(fc2_2).view(x_support_feature_1_2.shape[0], x_support_feature_1_2.shape[1], 1, 1)
            x_support_feature_se_1_2 = x_support_feature_4_1_2 * fc2_2
            x_support_feature_se_1_2 = F.adaptive_avg_pool2d(x_support_feature_se_1_2, 1).view(x_support_feature_1_2.shape[0], x_support_feature_1_2.shape[1])

            p1 = F.linear(x_support_feature_se_1_1, p) + x_support_feature_se_1_1
            x_support_feature_se_1_2.detach()
            loss_contrast = -(F.cosine_similarity(p1,x_support_feature_se_1_2).mean())

            x_support_feature_se_1_1 = x_support_feature_se_1_1.t()
            x_support_feature_se_1_1 = F.normalize(x_support_feature_se_1_1, dim=0)
            x_query_feature_recover_1 = torch.mm(x_support_feature_se_1_1, w)
            loss_s_r = torch.sum(torch.pow(x_query_feature_1-x_query_feature_recover_1, 2))
            loss = loss_s_r + loss_contrast * balance
            g = torch.autograd.grad(loss, [w1,w2,p,w])
            w1 = w1 - hebb_lr * g[0]
            w2 = w2 - hebb_lr * g[1]
            p = p - hebb_lr * g[2]
            w = w - hebb_lr * g[3]


        _, x_support_feature_1, x_support_feature_4_1 = [], [], []
        with torch.no_grad():
            for i in range(int((n_shot*k_way) / 5)):
                __, x_support_feature_, x_support_feature_4_ = model(x_support[i * 5:i * 5 + 5], output_layer=False,
                                                                     return_feature=True,
                                                                     addse=True)
                _.append(__)
                x_support_feature_1.append(x_support_feature_)
                x_support_feature_4_1.append(x_support_feature_4_)
            _, x_support_feature_1, x_support_feature_4_1 = torch.cat(_), torch.cat(x_support_feature_1), torch.cat(
                x_support_feature_4_1)

        fc1 = F.linear(x_support_feature_1, w1)
        fc1 = F.relu(fc1)
        fc2 = F.linear(fc1, w2)
        fc2 = F.sigmoid(fc2).view(x_support_feature_1.shape[0], x_support_feature_1.shape[1], 1, 1)
        x_support_feature_se_1 = x_support_feature_4_1 * fc2
        x_support_feature_se_1 = F.adaptive_avg_pool2d(x_support_feature_se_1, 1).view(x_support_feature_1.shape[0],
                                                                                   x_support_feature_1.shape[1])
        x_support_feature_se_1 = x_support_feature_se_1.t()
        x_support_feature_se_1 = F.normalize(x_support_feature_se_1, dim=0)

        x_query_recover = []
        for cls in range(k_way):
            x_query_recover_cls = torch.mm(x_support_feature_se_1[:, cls * n_shot:cls * n_shot + n_shot],
                                               w[cls * n_shot:cls * n_shot + n_shot, :])
            x_query_recover.append(x_query_recover_cls)
            pass

        x_query_recover = torch.cat(x_query_recover).reshape(k_way, -1, k_way*q_queries)
        x_query_feature_gt = x_query_feature_1.unsqueeze(0).repeat(k_way, 1,1)
        res = torch.pow((x_query_recover - x_query_feature_gt),2).sum(1)
        y_query_pred = torch.argmin(res, dim=0)
        task_predictions.append(y_query_pred)
        task_losses.append(loss)
    meta_batch_loss = torch.stack(task_losses).mean()
    return meta_batch_loss, torch.cat(task_predictions)
