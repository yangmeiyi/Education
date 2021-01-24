# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ActionModel.StoreMemory import Memory
from Userfrature_GAN.GeneratorModel import GeneratorModel
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

class GeneratorPolicy(nn.Module):
    def __init__(self, dim_user_state=70, dim_hidden1=128, dim_hidden2=256, dim_user_action=12, activation=nn.LeakyReLU):
        super(GeneratorPolicy, self).__init__()
        self.UserModel = GeneratorModel()
        self.UserModel.to(device).load()
        self.UserPolicy = nn.Sequential(
            nn.Linear(dim_user_state, dim_hidden1),
            activation(),
            nn.Linear(dim_hidden1, dim_hidden2),
            activation(),
            nn.Linear(dim_hidden2, dim_user_action)
        )
        self.UserPolicy.apply(init_weight)
        self.memory = Memory()
        to_device(self.UserPolicy)

    def forward(self, x):
        x = self.UserPolicy(x)
        x = F.softmax(x)
        return x


    # user_state
    def get_user_action(self, user_state):
        action = self.forward(user_state)
        return action



    def generate_batch(self, mini_batch_size=5000):
        """generate enough (state, action) pairs into memory, at least min_batch_size items.
        """
        self.memory.clear()

        num_items = 0  # count generated (state, action) pairs

        while num_items < mini_batch_size:
            # sample user and action from GAN
            state = self.UserModel.generate()
            action = self.get_user_action(state)

            # add to memory
            self.memory.push(state.detach().cpu().numpy(), action.detach().cpu().numpy())
            num_items += 1

    def sample_batch(self, batch_size):
        # sample batch (state, action) pairs from memory
        batch = self.memory.sample(batch_size)

        batch_state = FLOAT(np.stack(batch.state)).squeeze(1).to(device)
        batch_action = LONG(np.stack(batch.action)).squeeze(1).to(device)

        assert batch_state.size(0) == batch_size, "Expected batch size (s,a) pairs"

        return batch_state, batch_action

    def get_log_prob(self, user_state):
        action = self.get_user_action(user_state)
        current_action = action
        return torch.log(current_action.unsqueeze(1))
