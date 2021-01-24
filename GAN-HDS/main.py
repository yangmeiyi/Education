# -*- coding: utf-8 -*-
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import json
from Userfrature_GAN.GAN_train import GanModel
from data_loader import load_dataset
from Utils import *


@click.command()
@click.option('--dataset_path', type=click.Path('r'), default='../Data.csv')
@click.option('--dim_seed', type=int, default=128, help='Seed dimension for Generator of GAN')
@click.option('--batch_size', type=int, default=128, help='Batch size for GAN-')
@click.option('--learning_rate_generator', 'lr_g', type=float, default=0.0005, help='Learning rate for Generator')
@click.option('--learning_rate_discriminator', 'lr_d', type=float, default=0.0001,
              help='Learning rate for Discriminator')
@click.option('--alpha', type=float, default=1.0, help='Coefficient for Entropy Loss')
@click.option('--beta', type=float, default=1.0, help='Coefficient for KL divergence')
@click.option('--seed', type=int, default=2019, help='Random seed for reproduce')



def train(dataset_path, dim_seed, batch_size, lr_g, lr_d, alpha, beta, seed):
    """
    Train GAN for generating real user features
    """
    dim_user = 70
    dim_seed = dim_seed
    expert_user_features = load_dataset(dataset_path)


    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = GanModel(dim_user, dim_seed, lr_g, lr_d, expert_user_features, batch_size=batch_size, alpha=alpha,
                       beta=beta).to(device)
    WL = model.train()
    model.save_model()

    state = {"WL": WL}
    torch.save(state, "./hyper_batch_256_G1r_0.0005_Dlr_0.0001.pth")

if __name__ == '__main__':
    train()

