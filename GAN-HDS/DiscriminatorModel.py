# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Utils import *

INT = torch.IntTensor
LONG = torch.LongTensor
BYTE = torch.ByteTensor
FLOAT = torch.FloatTensor

def init_weight(m):
    if type(m) == nn.Linear:
        size = m.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.fill_(0.0)

class DiscriminatorModel(nn.Module):
    def __init__(self, dim_user=70, dim_hidden=256, dim_out=1, activation=nn.LeakyReLU):
        super(DiscriminatorModel, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(dim_user, dim_hidden),
            activation(),
            nn.Linear(dim_hidden, dim_out),
            nn.Sigmoid()
        )
        self.discriminator.apply(init_weight)

    def forward(self, x):
        return self.discriminator(x)
