import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.building_blocks import MLP, Conv, DeConv
from typing import Optional, Dict, List, Tuple


class OptionPolicy(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        input_dim: int,
        fc_dim: int,
        a_dim: int,
        num_options: int,
        activation: nn.Module = nn.Tanh(),
    ):
        super(OptionPolicy, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self._a_dim = a_dim
        self._dtype = torch.float32
        self._num_options = num_options

        self._max_val = 0.6
        self._min_val = 0.1

        self.models = nn.ModuleList()
        for _ in range(num_options):
            self.models.append(self.create_sequential_model(input_dim, fc_dim, a_dim))

    def create_sequential_model(self, input_dim, fc_dim, output_dim):
        return MLP(input_dim, (fc_dim, fc_dim), output_dim, activation=self.act)

    def forward(self, x: torch.Tensor, z: int, deterministic=False):
        logits = self.models[z](x)
        # implement std for cat distribution

        probs = F.softmax(logits, dim=-1)
        p_probs = torch.clamp(probs, min=self._min_val, max=self._max_val)
        probs = (p_probs - self._min_val) / (self._max_val - self._min_val)

        logprobs = torch.log(probs)

        dist = torch.distributions.categorical.Categorical(probs)
        if deterministic:
            # [N, |O|]
            a = torch.argmax(probs, dim=-1).to(self._dtype)
        else:
            a = dist.sample()
        a_oh = F.one_hot(
            a.long(),
            num_classes=self._a_dim,
        )
        probs = torch.sum(probs * a_oh, axis=-1, keepdim=True)
        logprobs = torch.sum(logprobs * a_oh, axis=-1, keepdim=True)

        return a, {
            "z": z,
            "logits": logits,
            "probs": probs,
            "logprobs": logprobs,
            "dist": dist,
        }


class OptionCritic(nn.Module):
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
        super(OptionCritic, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self._dtype = torch.float32

        # ex_layer = self.create_sequential_model(sf_dim, fc_dim, 1)

        self.models = nn.ModuleList()
        for _ in range(num_options):
            self.models.append(self.create_sequential_model(input_dim, fc_dim, 1))

    def create_sequential_model(self, input_dim, fc_dim, output_dim):
        return MLP(input_dim, (fc_dim, fc_dim), output_dim, activation=self.act)

    def forward(self, x: torch.Tensor, z: int):
        value = self.models[z](x)
        # for name, param in self.models[z].named_parameters():
        #     if "weight" in name:
        #         print(torch.norm(param.data, p=2).detach().cpu().numpy(), end=" ")
        #     if "bias" in name:
        #         print(torch.norm(param.data, p=2).detach().cpu().numpy(), end=" ")
        # print(torch.norm(value, p=2).clone().detach().cpu().numpy())

        return value, {"z": z}


class PsiAdvantage2(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self, fc_dim: int, sf_dim: int, a_dim: int, activation: nn.Module = nn.ReLU()
    ):
        super(PsiAdvantage2, self).__init__()

        # |A| duplicate networks
        self.act = activation
        # ex_layer = self.create_sequential_model(fc_dim, sf_dim)

        self.models = nn.ModuleList()
        for _ in range(a_dim):
            self.models.append(self.create_sequential_model(fc_dim, sf_dim))

    def create_sequential_model(self, fc_dim, sf_dim):
        return MLP(sf_dim, (fc_dim, fc_dim), sf_dim, activation=self.act)

    def forward(self, x: torch.Tensor):
        X = []
        for model in self.models:
            X.append(model(x.clone()))
        X = torch.stack(X, dim=1)

        return X


class PsiCritic2(nn.Module):
    """
    s
    """

    def __init__(
        self,
        fc_dim: int,
        sf_dim: int,
        a_dim: int,
        num_options: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super(PsiCritic2, self).__init__()

        # Algorithmic parameters
        self.act = activation
        self._a_dim = a_dim
        self._dtype = torch.float32
        self._num_options = num_options

        self.psi_advantages = nn.ModuleList()
        for _ in range(num_options):
            self.psi_advantages.append(PsiAdvantage2(fc_dim, sf_dim, a_dim, self.act))

        self.psi_states = nn.ModuleList()
        for _ in range(num_options):
            self.psi_states.append(
                MLP(
                    input_dim=sf_dim,
                    hidden_dims=(fc_dim, fc_dim),
                    output_dim=sf_dim,
                    activation=self.act,
                )
            )

    def forward(self, x: torch.Tensor, z: int):
        """
        x: phi
        phi = (phi_r, phi_s)
        psi = (psi_r, psi_s)
        Q = psi_r * w where w = eig(psi_s)
        ------------------------------------------
        Previous method of Q = psi_s * w where w = eig(psi_s) aims to navigate to 'bottleneck' while that may not be a goal
        therefore we need to modify the Q-direction by projecting onto the reward space.
        """
        psi_advantage = self.psi_advantages[z](x)
        psi_state = self.psi_states[z](x)

        psi = (
            psi_state.unsqueeze(1)
            + psi_advantage
            - torch.mean(psi_advantage, axis=1, keepdim=True)
        )  # psi ~ [N, |A|, F]

        return psi, {"psiState": psi_state, "psiAdvantage": psi_advantage}
