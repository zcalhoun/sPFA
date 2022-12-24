from .base import BaseModel

import torch
from torch import nn

import pdb


class DeepEncoder(BaseModel):
    __slots__ = [
        "vocab",
        "num_components",
        "hidden_size",
        "prior_mean",
        "prior_logvar",
        "enc_mu",
        "enc_logvar",
        "W_tilde",
        "pois_nll",
        "softplus",
        "beta",
        "device",
        "fc1",
    ]

    def __init__(
        self,
        hidden_size=1000,
        vocab=None,
        num_components=50,
        prior_mean=0,
        prior_logvar=0,
        beta_bias=50.0,
        device="auto",
    ):
        super(DeepEncoder, self).__init__(
            vocab=vocab,
            num_components=num_components,
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
            beta_bias=beta_bias,
            device=device,
        )
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.vocab, self.hidden_size, bias=True)

        self.enc_mu = nn.Linear(hidden_size, self.num_components, bias=True)
        self.enc_logvar = nn.Linear(hidden_size, self.num_components, bias=True)

        # Initialize the params
        with torch.no_grad():
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.enc_mu.weight)
            nn.init.xavier_uniform_(self.enc_logvar.weight)

    # Redefine the encoder
    def encode(self, x):
        x = self.softplus(self.fc1(x))
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        return mu, logvar
