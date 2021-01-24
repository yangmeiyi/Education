from Noise import OUNoise
from Utils import *


class DiscriminatorModel(nn.Module):
    def __init__(self, n_input=70 + 12, n_hidden=256, n_output=1, activation=nn.LeakyReLU):  # input = state+action
        super(DiscriminatorModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            activation(),
            nn.Linear(n_hidden, n_output),
            nn.Dropout(p=0.6),
            nn.Sigmoid()
        )

        self.Noise = OUNoise(n_input)
        self.model.apply(init_weight)

    def forward(self, x):
        # noise = torch.zeros_like(x)
        # for i in range(noise.size(0)):
        #     noise[i] += FLOAT(self.Noise.sample()).to(device)
        # x += noise
        # print(x.size())
        return self.model(x)
