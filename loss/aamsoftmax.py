#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import accuracy
from torch.jit import script

class aamsoftmax(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30, device="cuda", **kwargs):
        super(aamsoftmax, self).__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.m = margin
        self.s = scale
        self.in_feats = embedding_dim
        self.W = torch.nn.Parameter(torch.randn(embedding_dim, num_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AAM-Softmax m=%.3f s=%.3f' % (self.m, self.s))
        print('Embedding dim is {}, number of speakers is {}'.format(embedding_dim, num_classes))

    # @script
    def forward(self, x, label=None):
        x = x.to(self.device)
        label = label.to(self.device)
        
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        # ㄴ 여기까지는 amsoftmax와 동일

        # Additive Angular Margin
        eps = 1e-7
        costh = costh.clamp(-1 + eps, 1 + eps)  # acos의 domain 문제 방지
        theta = torch.acos(costh)  # cos^-1 적용하여 각도로 변환
        costh_m = torch.cos(theta + self.m)  # Margin을 각도에 더한 후 cosine 변환
        
        # one-hot encoding 방식
        one_hot = torch.zeros_like(costh).to(self.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        # 수정된 코사인 값 계산
        output = costh * (1 - one_hot) + costh_m * one_hot
        output = output * self.s
        
        loss = self.ce(output, label)
        acc = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, acc