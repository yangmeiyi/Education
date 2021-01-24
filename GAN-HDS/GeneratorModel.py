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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def init_weight(m):
    if type(m) == nn.Linear:
        size = m.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.fill_(0.0)

class GeneratorModel(nn.Module):
    def __init__(self, dim_user=70, dim_seed=128, dim_hidden=256, activation=nn.LeakyReLU):
        super(GeneratorModel, self).__init__()
        self.dim_seed = dim_seed
        self.generator = nn.Sequential(
            nn.Linear(dim_seed, dim_hidden),
            activation(),
            nn.Linear(dim_hidden, dim_user),
            nn.Tanh()
        )
        self.generator.apply(init_weight)

    def forward(self, z):
        return self.generator(z)

    # generate user sample from random seed z
    def generate(self, z=None):
        if z is None:
            z = torch.rand((1, self.dim_seed)).to(device)  # generate 1 random seed
        x = self.get_prob_entropy(self.generator(z))[0]  # softmax_feature
        features = [None] * 20
        features[0] = x[:, :3]
        features[1] = x[:, 3:4]
        features[2] = x[:, 4:5]
        features[3] = x[:, 5:17]
        features[4] = x[:, 17:37]
        features[5] = x[:, 37:38]
        features[6] = x[:, 38:40]
        features[7] = x[:, 40:41]
        features[8] = x[:, 41:42]
        features[9] = x[:, 42:43]
        features[10] = x[:, 43:44]
        features[11] = x[:, 44:46]
        features[12] = x[:, 46:47]
        features[13] = x[:, 47:49]
        features[14] = x[:, 49:50]
        features[15] = x[:, 50:54]
        features[16] = x[:, 54:61]
        features[17] = x[:, 61:68]
        features[18] = x[:, 68:69]
        features[19] = x[:, 69:70]

        one_hot = FLOAT([]).to(device)
        for i in range(20):
            tmp = torch.zeros_like(features[i], device=device)
            if (i >= 1 and i <= 2) or (i >= 5 and i <= 9) or (i >= 18 and i <= 19):
                one_hot = torch.cat((one_hot,  features[i]), dim=-1)
            else:
                one_hot = torch.cat((one_hot, tmp.scatter_(1, torch.multinomial(features[i], 1), 1)), dim=-1)
        return one_hot
        # return one_hot, x

    def get_prob_entropy(self, x):
        features = [None] * 20
        features[0] = x[:, :3]
        features[1] = x[:, 3:4]
        features[2] = x[:, 4:5]
        features[3] = x[:, 5:17]
        features[4] = x[:, 17:37]
        features[5] = x[:, 37:38]
        features[6] = x[:, 38:40]
        features[7] = x[:, 40:41]
        features[8] = x[:, 41:42]
        features[9] = x[:, 42:43]
        features[10] = x[:, 43:44]
        features[11] = x[:, 44:46]
        features[12] = x[:, 46:47]
        features[13] = x[:, 47:49]
        features[14] = x[:, 49:50]
        features[15] = x[:, 50:54]
        features[16] = x[:, 54:61]
        features[17] = x[:, 61:68]
        features[18] = x[:, 68:69]
        features[19] = x[:, 69:70]
        entropy = 0.0
        softmax_feature = FLOAT([]).to(device)
        for i in range(20):
            softmax_feature = torch.cat([softmax_feature.to(device), F.softmax(features[i].to(device), dim=1)], dim=-1)
            entropy += -(F.log_softmax(features[i].to(device), dim=1) * F.softmax(features[i].to(device), dim=1)).sum(dim=1).mean()
        return softmax_feature, entropy

    def load(self, path=None):
        if path is None:
            path = r'../user_G.pt'
        self.load_state_dict(torch.load(path))

