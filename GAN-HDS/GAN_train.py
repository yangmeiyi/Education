# -*- coding: utf-8 -*-

from Userfrature_GAN.DiscriminatorModel import DiscriminatorModel
from Userfrature_GAN.GeneratorModel import GeneratorModel
from Utils import *


class GanModel(nn.Module):
    def __init__(self, dim_user, dim_seed, lr_g, lr_d, expert_users, batch_size, alpha=1.0, beta=1.0):
        super(GanModel, self).__init__()

        self.dim_user = dim_user
        self.dim_seed = dim_seed


        # self.expert_users = self.dim_expand(expert_users[:, 1:70])
        self.expert_users = expert_users[:, 1:71]
        self.expert_users = torch.tensor(self.expert_users, dtype=torch.float32).to(device)
        self.batch_size = batch_size
        self.n_expert_users = self.expert_users.size(0)  # 100000



        self.alpha = alpha
        self.beta = beta

        self.G = GeneratorModel(activation=nn.Tanh)
        self.D = DiscriminatorModel(activation=nn.LeakyReLU)

        self.optim_G = optim.Adam(self.G.parameters(), lr=lr_g)
        self.optim_D = optim.Adam(self.D.parameters(), lr=lr_d)
        self.loss_func = nn.BCELoss()

        to_device(self.G, self.D, self.loss_func)

    @log_func("Train GANModel")
    def train(self):
        W_distance = []
        KL = []
        n_batch = (self.n_expert_users + self.batch_size - 1) // self.batch_size

        time_start = time.time()
        writer = SummaryWriter()

        for epoch in range(1000):
            wl_epoch = []
            kl_epoch = []
            idx = torch.randperm(self.n_expert_users)
            for i in range(n_batch):

                batch_expert = self.expert_users[idx[i * self.batch_size:(i + 1) * self.batch_size]]

                # sample minibatch from generator
                batch_seed = torch.normal(torch.zeros(batch_expert.size(0), self.dim_seed),
                                          torch.ones(batch_expert.size(0), self.dim_seed)).to(device)
                batch_gen = self.G.generate(batch_seed)


                # gradient ascent update discriminator
                for _ in range(1):
                    self.optim_D.zero_grad()
                    expert_data = self.D(batch_expert.to(device))
                    # gen_data = self.D(batch_gen.detach())
                    gen_data = self.D(batch_gen)

                    '''
                    # the loss of GAN #
                    d1 = self.loss_func(expert_data, torch.ones_like(expert_data, device=device))  # ture
                    d2 = self.loss_func(gen_data, torch.zeros_like(gen_data, device=device))  # false
                    d_loss = d1 + d2
                    self.optim_D.step()
                    '''

                    # the loss of WGAN #
                    d_loss = - self.get_w(expert_data, gen_data)  # maximize W distance
                    # d_loss.backward()
                    self.optim_D.step()

                # gradient ascent update generator
                for _ in range(10):
                    self.optim_G.zero_grad()
                    # sample minibatch from generator
                    batch_seed = torch.normal(torch.zeros(batch_expert.size(0), self.dim_seed),
                                              torch.ones(batch_expert.size(0), self.dim_seed)).to(device)
                    batch_gen = self.G.generate(batch_seed)
                    # feature = torch.tensor(batch_gen, dtype=torch.float32)
                    gen_data = self.D(batch_gen.detach())

                    # kl = self.get_kl(batch_gen, batch_expert)
                    # kl_epoch.append(kl.cpu().detach().mean().numpy())

                    wl = self.w_distance(batch_expert, batch_gen)
                    wl_epoch.append(wl.cpu().detach().mean().numpy())

                    g_loss =self.beta * wl - gen_data.mean() - self.alpha * self.G.get_prob_entropy(batch_gen)[1]
                    g_loss.backward()
                    self.optim_G.step()


                writer.add_scalars('GAN_SD/train_loss', {'discriminator_GAN_SD': d_loss,
                                                         'generator_GAN_SD': g_loss,
                                                         'W_distance': wl},
                                   epoch * n_batch + i)

                if i % 10 == 0:
                    cur_time = time.time() - time_start
                    eta = cur_time / (i + 1) * (n_batch - i - 1)
                    # print("Epoch: {}, Batch: {}, G_loss: {}, D_loss: {},  KL:{}, WL: {}, Time: {}, ETA: {}".format(
                    #     epoch, i, g_loss.data, d_loss.data, kl.data, wl.data, cur_time, eta))
            if (epoch + 1) % 50 == 0:
                self.save_model()
            # KL.append(np.mean(kl_epoch))
            W_distance.append(np.mean(wl_epoch))
        return W_distance



    def get_w(self, batch_expert, batch_gen_feature):
        return torch.mean(batch_expert) - torch.mean(batch_gen_feature)

    # the distance of two data distributions
    def w_distance(self, real, fake):
        return torch.mean(torch.abs(real - fake))



    def get_kl(self, batch_gen_feature, batch_expert):
        batch_gen = [None] * 20
        batch_gen[0] = batch_gen_feature[:, :3]
        batch_gen[1] = batch_gen_feature[:, 3:4]
        batch_gen[2] = batch_gen_feature[:, 4:5]
        batch_gen[3] = batch_gen_feature[:, 5:17]
        batch_gen[4] = batch_gen_feature[:, 17:37]
        batch_gen[5] = batch_gen_feature[:, 37:38]
        batch_gen[6] = batch_gen_feature[:, 38:40]
        batch_gen[7] = batch_gen_feature[:, 40:41]
        batch_gen[8] = batch_gen_feature[:, 41:42]
        batch_gen[9] = batch_gen_feature[:, 42:43]
        batch_gen[10] = batch_gen_feature[:, 43:44]
        batch_gen[11] = batch_gen_feature[:, 44:46]
        batch_gen[12] = batch_gen_feature[:, 46:47]
        batch_gen[13] = batch_gen_feature[:, 47:49]
        batch_gen[14] = batch_gen_feature[:, 49:50]
        batch_gen[15] = batch_gen_feature[:, 50:54]
        batch_gen[16] = batch_gen_feature[:, 54:61]
        batch_gen[17] = batch_gen_feature[:, 61:68]
        batch_gen[18] = batch_gen_feature[:, 68:69]
        batch_gen[19] = batch_gen_feature[:, 69:70]
        distributions = [None] * 20
        distributions[0] = batch_expert[:, :3]
        distributions[1] = batch_expert[:, 3:4]
        distributions[2] = batch_expert[:, 4:5]
        distributions[3] = batch_expert[:, 5:17]
        distributions[4] = batch_expert[:, 17:37]
        distributions[5] = batch_expert[:, 37:38]
        distributions[6] = batch_expert[:, 38:40]
        distributions[7] = batch_expert[:, 40:41]
        distributions[8] = batch_expert[:, 41:42]
        distributions[9] = batch_expert[:, 42:43]
        distributions[10] = batch_expert[:, 43:44]
        distributions[11] = batch_expert[:, 44:46]
        distributions[12] = batch_expert[:, 46:47]
        distributions[13] = batch_expert[:, 47:49]
        distributions[14] = batch_expert[:, 49:50]
        distributions[15] = batch_expert[:, 50:54]
        distributions[16] = batch_expert[:, 54:61]
        distributions[17] = batch_expert[:, 61:68]
        distributions[18] = batch_expert[:, 68:69]
        distributions[19] = batch_expert[:, 69:70]

        kl = 0.0
        for i in range(20):
            kl += (F.softmax(batch_gen[i].to(device), dim=1) *
                   (F.log_softmax(batch_gen[i].to(device), dim=1) -
                    F.log_softmax(distributions[i].to(device), dim=1))).sum(dim=1).mean()

        return kl


    def save_model(self):
        torch.save(self.G.state_dict(), r'../user_G.pt')
        torch.save(self.D.state_dict(), r'../user_D.pt')

