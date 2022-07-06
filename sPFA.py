"""
Supervised Poisson Factor Analysis
"""

import torch
from torch import nn
from torch.nn import functional as F


class sPFA(nn.Module):
    """
    sPFA - Supervised Poisson Factor Analysis
    """

    def __init__(
        self,
        vocab,
        num_components=20,
        prior_mean=0,
        prior_logvar=0,
        l1_reg=0.0,
        dropout=0.5,
        pred_dropout=0.5,
        device="auto",
    ):
        """
        Inputs
        --------
        vocab<int>: the size of the vocabulary

        This model only has the variational layer, then the output
        to the reconstruction. At this point, there are no hidden layers.
        """
        super().__init__()
        self.num_components = num_components
        self.prior_mean = torch.tensor(prior_mean)
        self.prior_logvar = torch.tensor(prior_logvar)
        self.l1_reg = l1_reg

        self.fc1 = nn.Linear(vocab, vocab)

        self.enc_mu = nn.Linear(vocab, num_components, bias=False)
        self.enc_logvar = nn.Linear(vocab, num_components, bias=False)

        self.W_tilde = nn.Parameter(torch.rand(num_components, vocab))

        self.pois_nll = nn.PoissonNLLLoss(log_input=False)
        self.softplus = nn.Softplus()

        self.dropout = nn.Dropout(dropout)
        #         self.lgamma = torch.lgamma
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
        return mu + eps * std

    def forward(self, x):

        # Randomly drop an input sometimes
        h1 = self.dropout(x)
        # Add in a hidden layer for more expressivity
        h2 = F.relu(self.fc1(h1))
        # Use the softplus here to ensure k and lam are
        # greater than 0.

        mu = self.enc_mu(h2)
        logvar = self.softplus(self.enc_logvar(h2))

        s_tilde = self.reparameterize(mu, logvar)

        s = self.softplus(s_tilde)
        W = F.relu(self.W_tilde)

        # Predict using dropout on the nodes coming
        # out of beta. This should prevent overfitting
        s_d = self.pred_dropout(s)
        y_hat = self.beta(s_d)

        return s, W, mu, logvar, y_hat

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

    def l1_loss(self):
        """This loss pushes down the beta parameters so that they are
        close to zero"""
        beta_params = list(self.beta.parameters())
        weights = torch.sum(torch.abs(beta_params[0]))
        intercept = torch.abs(beta_params[1])
        return self.l1_reg * (weights + intercept)

    def loss_function(self, recon_x, x, mu, logvar, y, y_hat):
        KLD = self._kl_divergence(mu, logvar)
        PNLL = self.pois_nll(recon_x, x)
        # This will disproportionately weight higher values of y
        MSE = (y - y_hat).pow(2).mean()

        #         # Add in the L1 loss
        #         L1 = 0
        #         for b in model.beta.parameters():
        #             L1 += b.abs().sum()

        return PNLL, MSE, KLD
