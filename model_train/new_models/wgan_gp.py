import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalMLPGenerator(nn.Module):
    def __init__(
        self,
        num_classes: int,
        z_dim: int = 128,
        emb_dim: int = 64,
        hidden: int = 512,
        out_len: int = 1500,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.out_len = out_len
        self.emb = nn.Embedding(num_classes, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + emb_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, out_len),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        e = self.emb(y)
        x = torch.cat([z, e], dim=1)
        out = self.net(x)
        return out


class ConditionalMLPCritic(nn.Module):
    def __init__(
        self,
        num_classes: int,
        emb_dim: int = 64,
        hidden: int = 512,
        in_len: int = 1500,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_classes, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(in_len + emb_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        e = self.emb(y)
        h = torch.cat([x, e], dim=1)
        return self.net(h).view(-1)


def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    b = real.size(0)
    alpha = torch.rand(b, 1, device=device)
    interp = alpha * real + (1 - alpha) * fake
    interp.requires_grad_(True)
    scores = critic(interp, y)
    grad = torch.autograd.grad(
        outputs=scores,
        inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(b, -1)
    gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
    return lambda_gp * gp


@torch.no_grad()
def sample_generator(
    gen: ConditionalMLPGenerator,
    num: int,
    num_classes: int,
    z_dim: int,
    device: torch.device,
    y: torch.Tensor,
) -> torch.Tensor:
    z = torch.randn(num, z_dim, device=device)
    y = y.to(device)
    x = gen(z, y)
    return x

