import click
import pandas as pd
import matplotlib.pyplot as plt
from ActionModel.GAN_train import GANModel
from data_loader import load_dataset
from Utils import *
import json

@click.command()
@click.option('--dataset_path', type=click.Path('r'), default='../Data.csv')
# @click.option('--batch_size', type=int, default=128, help='Batch size for GAN-SD')
# @click.option('--learning_rate_generator', 'lr_g', type=float, default=0.001, help='Learning rate for Generator')
# @click.option('--learning_rate_discriminator', 'lr_d', type=float, default=0.0001,
#               help='Learning rate for Discriminator')
@click.option('--seed', type=int, default=2019, help='Random seed for reproduce')
def main(dataset_path, seed):

    expert_user_trajectory = load_dataset(dataset_path)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = GANModel(expert_user_trajectory)
    Reward = model.train()

    with open("./hyper_batch_256_D1r_0.0005_5000.json", "w") as f:
        json.dump(Reward, f)


if __name__ == '__main__':
    main()
