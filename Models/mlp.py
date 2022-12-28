import torch
from torch import nn
from torch.nn import functional as F


class mlp(nn.Module):
    __slots__ = ["hidden_1", "hidden_2", "fc1", "fc2", "device"]

    def __init__(self, hidden_1, hidden_2):
        super(mlp, self).__init__()
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.fc1 = nn.Linear(self.hidden_1, self.hidden_2, bias=True)
        self.fc2 = nn.Linear(self.hidden_2, 1, bias=True)

        # Set the device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        # Initialize the params
        with torch.no_grad():
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def compute_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)
