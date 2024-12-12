import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from models.layers.building_blocks import MLP


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
        # |A| duplicate networks
        self.act = activation

        self._a_dim = num_options + 1
        self._dtype = torch.float32
        self._num_options = num_options

        self.model = MLP(
            input_dim, (fc_dim, fc_dim, fc_dim), self._a_dim, activation=self.act
        )

    def forward(self, state: torch.Tensor, deterministic=False):
        # when the input is raw by forawrd() not learn()
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
            state = state.reshape(state.shape[0], -1)

        logits = self.model(state)

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        z_argmax = (
            torch.argmax(probs, dim=-1).long()
            if deterministic
            else dist.sample().long()
        )
        z = F.one_hot(z_argmax.long(), num_classes=self._a_dim)

        logprobs = dist.log_prob(z_argmax)
        probs = torch.sum(probs * z, dim=-1)
        entropy = dist.entropy()

        return (
            z,
            z_argmax,
            {"dist": dist, "probs": probs, "logprobs": logprobs, "entropy": entropy},
        )

    def log_prob(self, dist: torch.distributions, actions: torch.Tensor):
        """
        Actions must be tensor
        """
        actions = actions.squeeze() if actions.shape[-1] > 1 else actions
        logprobs = dist.log_prob(torch.argmax(actions, dim=-1)).unsqueeze(-1)
        return logprobs

    def entropy(self, dist: torch.distributions):
        """
        For code consistency
        """
        return dist.entropy().unsqueeze(-1)


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

        dist = Categorical(probs)

        if deterministic:
            # convert to long for indexing purpose
            z = torch.argmax(probs, dim=-1).long()
        else:
            z = dist.sample().long()

        logprobs = dist.log_prob(z)
        probs = torch.argmax(probs, dim=-1)

        return z, {"logits": logits, "probs": probs, "logprobs": logprobs}
