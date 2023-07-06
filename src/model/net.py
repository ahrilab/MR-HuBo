import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(dim_input, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_output)

        self.acti = nn.GELU()

    def forward(self, inp):
        out = self.acti(self.fc1(inp))
        out = self.acti(self.fc2(out))
        out = self.fc3(out)

        return out
