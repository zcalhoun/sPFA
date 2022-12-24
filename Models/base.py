import torch
from torch import nn
from torch.nn import functional as F


class BaseModel(nn.Module):
    __slots__ = [
        "vocab",
        "num_components",
        "prior_mean",
        "prior_logvar",
        "enc_mu",
        "enc_logvar",
        "W_tilde",
        "pois_nll",
        "softplus",
        "beta",
        "device",
    ]

    def __init__(
        self,
        vocab=None,
        num_components=50,
        prior_mean=0,
        prior_logvar=0,
        beta_bias=50.0,
        device="auto",
    ):
        super(BaseModel, self).__init__()
        self.vocab = vocab
        self.num_components = num_components
        self.prior_mean = torch.tensor(prior_mean)
        self.prior_logvar = torch.tensor(prior_logvar)

        self.enc_mu = nn.Linear(vocab, num_components, bias=False)
        self.enc_logvar = nn.Linear(vocab, num_components, bias=False)

        self.W_tilde = nn.Parameter(torch.rand(num_components, vocab))

        with torch.no_grad():
            nn.init.trunc_normal_(self.W_tilde, mean=-2, std=1, a=-2, b=2)

        self.pois_nll = nn.PoissonNLLLoss(log_input=False)
        self.softplus = nn.Softplus()

        self.beta = Beta(num_components, bias=beta_bias)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.to(self.device)

        # Initialize the params
        with torch.no_grad():
            nn.init.xavier_uniform_(self.enc_mu.weight)
            nn.init.xavier_uniform_(self.enc_logvar.weight)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        return mu, logvar

    def decode(self, s):
        W = self.softplus(self.W_tilde)
        x_hat = torch.matmul(s, W)
        return x_hat

    def predict_aqi(self, s):
        return self.beta(s)

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

    def _kl_divergence(self, mu, logvar):

        kld = torch.mean(
            -0.5
            * torch.sum(
                1
                + logvar
                - self.prior_logvar
                - (mu - self.prior_mean) ** 2 / self.prior_logvar.exp()
                - logvar.exp() / self.prior_logvar.exp(),
                dim=1,
            ),
            dim=0,
        )

        # If no prior
        # kld = torch.mean(
        #     -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0
        # )

        return kld

    def compute_loss(self, x, x_hat, y, y_hat, mu, logvar, w):
        # reconstruction loss
        recon_loss = self.pois_nll(x_hat, x)

        # KL divergence
        kld = self._kl_divergence(mu, logvar)

        # AQI loss
        aqi_loss = (w * (y - y_hat).pow(2)).mean()

        return recon_loss, kld, aqi_loss


class Beta(nn.Module):
    """This layer ensures that the beta parameter is positive"""

    def __init__(self, topics: int, bias: float) -> None:
        super(Beta, self).__init__()
        self.beta = nn.Parameter(torch.rand(topics))
        self.bias = nn.Parameter(torch.Tensor([bias]))
        self.softplus = nn.Softplus()

        with torch.no_grad():
            nn.init.trunc_normal_(self.beta, mean=-2, std=1, a=-2, b=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.softplus(self.beta) + self.bias
