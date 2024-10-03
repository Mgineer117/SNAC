import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.building_blocks import MLP, Conv, DeConv
from typing import Optional, Dict, List, Tuple


class PPO_Policy(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self, input_dim: int, fc_dim: int, a_dim: int, activation: nn.Module = nn.ReLU()
    ):
        super(PPO_Policy, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self._a_dim = a_dim
        self._dtype = torch.float32

        self.model = MLP(input_dim, (fc_dim, fc_dim), a_dim, activation=self.act)

    def forward(self, x: torch.Tensor):
        logits = self.model(x)

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.categorical.Categorical(probs)
        return dist


class PPO_Critic(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(self, input_dim: int, fc_dim: int, activation: nn.Module = nn.ReLU()):
        super(PPO_Critic, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self._dtype = torch.float32

        self.model = MLP(input_dim, (fc_dim, fc_dim), 1, activation=self.act)

    def forward(self, x: torch.Tensor):
        value = self.model(x)
        return value
