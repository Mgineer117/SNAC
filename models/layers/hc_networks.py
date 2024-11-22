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
        num_options += 1  # for primitive actions
        # |A| duplicate networks
        self.act = activation

        self._dtype = torch.float32
        self._num_options = num_options

        self.dist = None

        self.model = MLP(input_dim, (fc_dim, fc_dim), num_options, activation=self.act)

        # parameters
        self._num_options = num_options

    def forward(self, state: torch.Tensor, deterministic=False):
        # when the input is raw by forawrd() not learn()
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
            state = state.reshape(state.shape[0], -1)

        logits = self.model(state)

        logprobs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(logprobs)
        dist = Categorical(probs)

        if deterministic:
            z = torch.argmax(probs, dim=-1).long()
        else:
            z = dist.sample().long()

        logprobs = dist.log_prob(z)
        probs = torch.exp(logprobs)

        self.dist = dist
        entropy = dist.entropy()

        return z, {
            "logits": logits,
            "entropy": entropy,
            "probs": probs,
            "logprobs": logprobs,
        }

    def log_prob(self, actions):
        return self.dist.log_prob(actions)


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
