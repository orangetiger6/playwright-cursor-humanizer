"""Autoregressive GRU with a bivariate Gaussian mixture head (Graves, 2013)."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .data import COND_DIM

LOG_2PI = math.log(2 * math.pi)


class MouseMDN(nn.Module):
    def __init__(self, cond_dim=COND_DIM, hidden=256, layers=2, mixtures=10, dropout=0.1):
        super().__init__()
        self.config = dict(cond_dim=cond_dim, hidden=hidden, layers=layers, mixtures=mixtures, dropout=dropout)
        self.mixtures = mixtures
        self.inp = nn.Sequential(nn.Linear(4 + cond_dim, hidden), nn.GELU())
        self.rnn = nn.GRU(hidden, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 6 * mixtures + 1)

    def forward(self, x, h=None):
        z, h = self.rnn(self.inp(x), h)
        return self._split(self.head(z)), h

    def _split(self, out):
        m = self.mixtures
        logit_pi, mu, log_sigma, rho, end = torch.split(out, [m, 2 * m, 2 * m, m, 1], dim=-1)
        shape = out.shape[:-1] + (m, 2)
        return {
            "logit_pi": logit_pi,
            "mu": mu.reshape(shape),
            "log_sigma": log_sigma.reshape(shape).clamp(-6.0, 3.0),
            "rho": torch.tanh(rho) * 0.95,
            "end_logit": end.squeeze(-1),
        }

    @staticmethod
    def loss(params, delta, end, mask):
        """Returns (total, delta NLL, end BCE), each averaged over valid steps."""
        z = (delta.unsqueeze(-2) - params["mu"]) / params["log_sigma"].exp()
        rho = params["rho"]
        one_minus = 1 - rho**2
        quad = (z[..., 0] ** 2 + z[..., 1] ** 2 - 2 * rho * z[..., 0] * z[..., 1]) / one_minus
        log_n = -0.5 * quad - LOG_2PI - params["log_sigma"].sum(-1) - 0.5 * torch.log(one_minus)
        nll = -torch.logsumexp(F.log_softmax(params["logit_pi"], -1) + log_n, -1)
        bce = F.binary_cross_entropy_with_logits(params["end_logit"], end, reduction="none")
        n = mask.sum()
        nll, bce = (nll * mask).sum() / n, (bce * mask).sum() / n
        return nll + bce, nll, bce

    @staticmethod
    @torch.no_grad()
    def sample_step(params, temperature=1.0, generator=None):
        """Sample one (delta, end) from single-step params (no batch/time dims)."""
        probs = F.softmax(params["logit_pi"] / temperature, -1)
        k = torch.multinomial(probs, 1, generator=generator).item()
        mu, rho = params["mu"][k], params["rho"][k]
        sigma = params["log_sigma"][k].exp() * math.sqrt(temperature)
        e = torch.randn(2, generator=generator)
        dx = mu[0] + sigma[0] * e[0]
        dy = mu[1] + sigma[1] * (rho * e[0] + torch.sqrt(1 - rho**2) * e[1])
        end = torch.rand(1, generator=generator).item() < torch.sigmoid(params["end_logit"]).item()
        return torch.stack([dx, dy]), end
