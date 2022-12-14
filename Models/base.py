import torch
from torch import nn
from torch.nn import functional as F


class BaseModel(nn.Module):
    def __init__(
        self,
        vocab=None,
        hidden=1000,
        num_components=50,
        prior_mean=0,
        prior_logvar=0,
        device="auto",
    ):
        super(BaseModel, self).__init__()
        self.num_components = num_components
        self.prior_mean = torch.tensor(prior_mean)
        self.prior_logvar = torch.tensor(prior_logvar)

        self.fc1 = nn.Linear(vocab, hidden)
        self.enc_mu = nn.Linear(hidden, num_components, bias=False)
        self.enc_logvar = nn.Linear(hidden, num_components, bias=False)

        self.W_tilde = nn.Parameter(torch.rand(num_components, vocab))

        self.pois_nll = nn.PoissonNLLLoss(log_input=False)
        self.softplus = nn.Softplus()

        self.pred_dropout = nn.Dropout(pred_dropout)

        self.beta = nn.Linear(num_components, 1, bias=True)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.to(self.device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        return mu, logvar

    def decode(self, s):
        W = self.softplus(self.W_tilde)
        x_hat = torch.matmul(s, W)
        return x_hat

    def predict_aqi(self, s):
        beta = self.softplus(self.beta)
        y_hat = torch.matmul(s, beta)
        return y_hat

    def forward(self, x):

        # encoding
        mu, logvar = self.encode(x)

        # reparameterization
        s = self.reparameterize(mu, logvar)
        s = self.softplus(s)

        # reconstruction
        x_hat = self.decode(s)
        y_hat = self.predict_aqi(s)

        return x_hat, y_hat, mu, logvar
