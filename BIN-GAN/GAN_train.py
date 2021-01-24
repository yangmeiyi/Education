from ActionModel.DiscriminatorModel import DiscriminatorModel
from ActionModel.GeneratorPolicy import GeneratorPolicy
from ActionModel.CriticModel import CriticModel
from ActionModel.ppo import GAE, PPO_step
from Utils import *
import seaborn as sns



class GANModel:
    def __init__(self, expert_data, lr_d=0.0005, lr_g=0.00001, lr_v=0.001, epochs=5000, batch_size=256,
                 ppo_epoch=16, epsilon=0.1, l2_reg=1e-3):
        self.expert_data = expert_data[:, 1:83]
        self.expert_data = torch.tensor(self.expert_data, dtype=torch.float32).to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.ppo_epoch = ppo_epoch
        self.epsilon = epsilon
        self.l2_reg = l2_reg
        self.dim_seed = 128

        self.D = DiscriminatorModel()
        self.G = GeneratorPolicy()  # generate enough (state, action) pairs into memory
        self.V = CriticModel()

        self.optim_G = optim.Adam(self.G.parameters(), lr=lr_g)
        self.optim_D = optim.Adam(self.D.parameters(), lr=lr_d)
        self.optim_V = optim.Adam(self.V.parameters(), lr=lr_v)

        self.loss_func = nn.BCELoss()
        self.grad_map = {}

        to_device(self.G, self.D, self.V, self.loss_func)

    def train(self):
        Reward = []
        writer = SummaryWriter()
        batch_num = (len(self.expert_data) + self.batch_size - 1) // self.batch_size
        iter_num = self.epochs * batch_num

        for epoch in range(self.epochs):
            # shuffle expert data
            idx = torch.randperm(len(self.expert_data))
            # generate (state, action) pairs
            gen_state_action_num = 10240
            self.G.generate_batch(gen_state_action_num)
            # sample generated (state, action) pairs from memory
            batch_gen_state, batch_gen_action = self.G.sample_batch(self.batch_size)

            ############################
            # update Discriminator
            ############################
            for i in range(batch_num):
                # sample a batch of expert trajectories
                batch_expert_old = self.expert_data[idx[i * self.batch_size:(i + 1) * self.batch_size]]


                # update Discriminator adaptively
                update_frequency = 1 if epoch < 10 else 20

                if i % update_frequency == 0:
                    self.optim_D.zero_grad()

                    expert_o = self.D(batch_expert_old.to(device))  # real
                    gen_o = self.D(
                        torch.cat([batch_gen_state, batch_gen_action.type(torch.float)], dim=1).to(device))  # fake


                    real_loss = self.loss_func(expert_o, torch.ones_like(expert_o, device=device))
                    fake_loss = self.loss_func(gen_o, torch.zeros_like(gen_o, device=device))
                    d_loss = real_loss + fake_loss


                    d_loss.register_hook(lambda grad: self.hook_grad("d_loss", grad))
                    d_loss.backward()

                    self.optim_D.step()

                print("=" * 100 + "Discriminator")
                print("Epoch: {}, Batch: {}, D_loss: {}, real_reward: {}, fake_reward: {}".format(
                    epoch, i, d_loss.data, expert_o.mean(), gen_o.mean()))

            with torch.no_grad():
                gen_r = self.D(torch.cat([batch_gen_state, batch_gen_action.type(torch.float)], dim=1).to(device))
                gen_r = gen_r.log()
                value_o = self.V(batch_gen_state)
                fixed_log_prob = self.G.get_log_prob(batch_gen_state)

            advantages, returns = GAE(- torch.log(1 - gen_r + 1e-6), value_o, gamma=0.95, lam=0.95)
            Reward.append(returns.cpu().mean().numpy()*(-1))

            ##############################
            # update Generator using PPO
            ##############################
            for k in range(15):
                new_index = torch.randperm(batch_gen_state.shape[0]).to(device)

                mini_batch_gen_state, mini_batch_gen_action, mini_batch_fixed_log_prob, mini_batch_returns, \
                mini_batch_advantages = batch_gen_state[new_index], batch_gen_action[new_index], fixed_log_prob[
                    new_index], returns[new_index], advantages[new_index]

                v_loss, p_loss = PPO_step(self.G, self.V, self.optim_G, self.optim_V, mini_batch_gen_state,
                                          mini_batch_gen_action, mini_batch_returns, mini_batch_advantages,
                                          mini_batch_fixed_log_prob, self.epsilon, self.l2_reg)

                print("=" * 100 + "Generator")
                print("Epoch: {}, Batch: {}, v_loss: {}, p_loss: {}".format(
                    epoch, i, v_loss.data, p_loss.data))
            self.save_model()
            torch.cuda.empty_cache()
        return Reward


    def w_distance(self, real, fake):
        return torch.mean(torch.abs(real - fake))


    def hook_grad(self, key, value):
        self.grad_map[key] = value

    def save_model(self):
        torch.save(self.G.state_dict(), r'../policy.pt')
        torch.save(self.V.state_dict(), r'../value.pt')
        torch.save(self.D.state_dict(), r'../discriminator.pt')
