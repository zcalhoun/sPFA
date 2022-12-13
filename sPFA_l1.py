"""
This script contains a refactored version of the sPFA script
that modularizes the code and simplifies the architecture
so that the model can leverage separate learning rates
for the encoder/decoder and the predictive layer
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class sPFA_L1(nn.Module):
    def __init__(
        self,
        vocab: int,
        topics: int,
        hidden1: int,
        hidden2: int,
        pred_dropout: float = 0.5,
        prior_mean: float = 0.0,
        prior_logvar: float = 0.0,
        batchnorm_eps: float = 0.001,
        batchnorm_momentum: float = 0.001,
        encoder_noise: float = 0.2,
        decoder_noise: float = 0.2,
        beta_bias: float = 50.0,
        device: str = "auto",
    ) -> None:
        super(sPFA_L1, self).__init__()

        self.topics = topics
        self.vocab = vocab
        self.prior_mean = torch.tensor(prior_mean)
        self.prior_logvar = torch.tensor(prior_logvar)

        # Define the architecture
        self.encoder = Encoder(
            vocab,
            topics,
            hidden1,
            hidden2,
            encoder_noise,
            batchnorm_eps,
            batchnorm_momentum,
        )
        self.decoder = Decoder(vocab, topics, decoder_noise)
        self.beta = Beta(topics, bias=beta_bias)

        # For the loss function
        self.pois_nll = nn.PoissonNLLLoss(log_input=False)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.to(self.device)

    def forward(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(batch)
        s = F.softplus(self.reparameterize(mu, logvar))

        recon = self.decoder(s)
        pred = self.beta(s)

        return recon, mu, logvar, pred, s

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kl_divergence(self, mean, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # BUT...
        # Code extended to handle a more informative prior
        # Referencing this derivation found here:
        # https://stanford.edu/~jduchi/projects/general_notes.pdf
        # Assume diagonal matrices for variance
        KLD = -0.5 * torch.sum(
            1
            + logvar
            - self.prior_logvar
            - (mean - self.prior_mean) ** 2 / self.prior_logvar.exp()
            - logvar.exp() / self.prior_logvar.exp()
        )

        return KLD

    def loss(self, recon_x, x, mu, logvar, y, y_hat, w):
        KLD = self._kl_divergence(mu, logvar)
        PNLL = self.pois_nll(recon_x, x)
        # w is the weight of the specific AQI value provided here.
        MSE = (w * (y - y_hat).pow(2)).mean()

        return PNLL, MSE, KLD


class Beta(nn.Module):
    """This layer ensures that the beta parameter is positive"""

    def __init__(self, topics: int, bias: float) -> None:
        super(Beta, self).__init__()
        self.beta = nn.Parameter(torch.rand(topics))
        self.bias = nn.Parameter(torch.Tensor([bias]))
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.softplus(self.beta) + self.bias


class Decoder(nn.Module):
    """
    Decoder
    """

    def __init__(self, vocab: int, topics: int, noise: float = 0.2) -> None:
        super(Decoder, self).__init__()
        self.W = nn.Parameter(torch.rand(topics, vocab))
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.softplus(self.W)
        return self.dropout(x @ W)


class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(
        self,
        vocab: int,
        topics: int,
        hidden1: int,
        hidden2: int,
        noise: float = 0.2,
        eps: float = 0.001,
        momentum: float = 0.001,
    ) -> None:
        super(Encoder, self).__init__()
        self.hidden = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vocab, hidden1)),
                    ("softplus1", nn.Softplus()),
                    ("linear2", nn.Linear(hidden1, hidden2)),
                    ("softplus2", nn.Softplus()),
                    ("dropout", nn.Dropout(noise)),
                ]
            )
        )

        self.mu = nn.Sequential(
            OrderedDict(
                [
                    ("linear", nn.Linear(hidden2, topics)),
                    (
                        "batchnorm",
                        nn.BatchNorm1d(topics, affine=True, eps=eps, momentum=momentum),
                    ),
                ]
            )
        )

        self.logvar = nn.Sequential(
            OrderedDict(
                [
                    ("linear", nn.Linear(hidden2, topics)),
                    (
                        "batchnorm",
                        nn.BatchNorm1d(topics, affine=True, eps=eps, momentum=momentum),
                    ),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.hidden(x)
        return self.mu(h1), self.logvar(h1)
