import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.building_blocks import MLP, Conv, DeConv
from typing import Optional, Dict, List, Tuple


class HC_Policy(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        input_dim: int,
        fc_dim: int,
        num_options: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super(HC_Policy, self).__init__()
        """
        a_dim must be num_options + 1
        """
        num_options += 1  # for primitive actions
        # |A| duplicate networks
        self.act = activation

        self._dtype = torch.float32
        self._num_options = num_options

        self._max_val = 0.6
        self._min_val = 0.1

        self.model = MLP(input_dim, (fc_dim, fc_dim), num_options, activation=self.act)

        # parameters
        self._num_options = num_options

    def forward(self, x: torch.Tensor, deterministic=False):
        logits = self.model(x)

        probs = F.softmax(logits, dim=-1)
        p_probs = torch.clamp(probs, min=self._min_val, max=self._max_val)
        probs = (p_probs - self._min_val) / (self._max_val - self._min_val)

        logprobs = torch.log(probs)

        # probs = F.softmax(logits, dim=-1) + 1e-7
        # logprobs = F.log_softmax(logits, dim=-1) + 1e-7

        dist = torch.distributions.categorical.Categorical(probs)

        if deterministic:
            z = torch.argmax(
                probs, dim=-1
            ).long()  # convert to long for indexing purpose
        else:
            z = dist.sample().long()
        z_oh = F.one_hot(z.long(), num_classes=self._num_options)
        probs = torch.sum(probs * z_oh, axis=-1, keepdim=True)
        logprobs = (
            torch.sum(logprobs * z_oh, axis=-1, keepdim=True) + 1e-7
        )  # for numeric stability
        return z, {
            "logits": logits,
            "probs": probs,
            "logprobs": logprobs,
            "dist": dist,
        }


class HC_PrimitivePolicy(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        input_dim: int,
        fc_dim: int,
        a_dim: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super(HC_PrimitivePolicy, self).__init__()
        """
        a_dim must be num_options + 1
        """
        # |A| duplicate networks
        self.act = activation
        self.model = MLP(input_dim, (fc_dim, fc_dim), a_dim, activation=self.act)

        # parameters
        self._a_dim = a_dim

    def forward(self, x: torch.Tensor, deterministic: bool = False):
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1) + 1e-7
        logprobs = F.log_softmax(logits, dim=-1) + 1e-7

        dist = torch.distributions.categorical.Categorical(probs)

        if deterministic:
            z = torch.argmax(
                probs, dim=-1
            ).long()  # convert to long for indexing purpose
        else:
            z = dist.sample().long()

        z_oh = F.one_hot(z.long(), num_classes=self._a_dim)

        probs = torch.sum(probs * z_oh, axis=-1, keepdim=True)
        logprobs = torch.sum(
            logprobs * z_oh, axis=-1, keepdim=True
        )  # for numeric stability

        return z, {"logits": logits, "probs": probs, "logprobs": logprobs}


class HC_Critic(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(self, input_dim: int, fc_dim: int, activation: nn.Module = nn.ReLU()):
        super(HC_Critic, self).__init__()

        # |A| duplicate networks
        self.act = activation
        self.model = MLP(input_dim, (fc_dim, fc_dim), 1, activation=self.act)

    def forward(self, x: torch.Tensor):
        value = self.model(x)
        return value, {}
