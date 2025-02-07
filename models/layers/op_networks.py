import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, Categorical
from models.layers.building_blocks import MLP


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
        is_discrete: bool = False,
    ):
        super(OptionPolicy, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self._a_dim = a_dim
        self._dtype = torch.float32
        self._num_options = num_options

        self.logstd = 0

        self.is_discrete = is_discrete

        self.models = nn.ModuleList()
        for _ in range(num_options):
            self.models.append(self.create_model(input_dim, fc_dim, a_dim))

    def create_model(self, input_dim, fc_dim, output_dim):
        return MLP(
            input_dim,
            (fc_dim, fc_dim),
            output_dim,
            activation=self.act,
            initialization="actor",
        )

    def forward(self, state: torch.Tensor, z: int, deterministic=False):
        # when the input is raw by forawrd() not learn()
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = state.reshape(state.shape[0], -1)

        raw_logits = self.models[z](state)

        if self.is_discrete:
            logits = F.softplus(raw_logits)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            a_argmax = torch.argmax(probs, dim=-1) if deterministic else dist.sample()
            a = F.one_hot(a_argmax.long(), num_classes=self._a_dim)

            logprobs = dist.log_prob(a_argmax).unsqueeze(-1)
            probs = torch.sum(probs * a, dim=-1)
        else:
            ### Shape the output as desired
            mu = F.tanh(raw_logits)
            std = torch.exp(self.logstd)

            covariance_matrix = torch.diag_embed(std**2)  # Variance is std^2
            dist = MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)

            a = mu if deterministic else dist.rsample()

            logprobs = dist.log_prob(a).unsqueeze(-1)
            probs = torch.exp(logprobs)

        entropy = dist.entropy()

        return a, {
            "z": z,
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def log_prob(self, dist: torch.distributions, actions: torch.Tensor):
        """
        Actions must be tensor
        """
        actions = actions.squeeze() if actions.shape[-1] > 1 else actions

        if self.is_discrete:
            logprobs = dist.log_prob(torch.argmax(actions, dim=-1)).unsqueeze(-1)
        else:
            logprobs = dist.log_prob(actions).unsqueeze(-1)
        return logprobs

    def entropy(self, dist: torch.distributions):
        """
        For code consistency
        """
        return dist.entropy().unsqueeze(-1)


class OP_Critic(nn.Module):
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
        super(OP_Critic, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self._dtype = torch.float32

        # ex_layer = self.create_sequential_model(sf_dim, fc_dim, 1)

        self.models = nn.ModuleList()
        for _ in range(num_options):
            self.models.append(self.create_model(input_dim, fc_dim, 1))

    def create_model(self, input_dim, fc_dim, output_dim):
        return MLP(
            input_dim,
            (fc_dim, fc_dim),
            output_dim,
            activation=self.act,
            initialization="critic",
        )

    def forward(self, state: torch.Tensor, z: int):
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = state.reshape(state.shape[0], -1)
        value = self.models[z](state)
        return value, {"z": z}


class OP_Q_Critic(nn.Module):
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
        super(OP_Q_Critic, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self._dtype = torch.float32

        # ex_layer = self.create_sequential_model(sf_dim, fc_dim, 1)

        self.models = nn.ModuleList()
        for _ in range(num_options):
            self.models.append(self.create_model(input_dim, fc_dim, 1))

    def create_model(self, input_dim, fc_dim, output_dim):
        return MLP(input_dim, (fc_dim, fc_dim), output_dim, activation=self.act)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, z: int):
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
            state = state.reshape(state.shape[0], -1)
        x = torch.cat([states, actions], dim=-1)
        value = self.models[z](x)
        return value


class OP_CriticTwin(nn.Module):
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
        super(OP_CriticTwin, self).__init__()

        # |A| duplicate networks
        self._dtype = torch.float32

        self.critic1 = OP_Q_Critic(input_dim, fc_dim, num_options, activation)
        self.critic2 = OP_Q_Critic(input_dim, fc_dim, num_options, activation)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, z: int):
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
            state = state.reshape(state.shape[0], -1)
        value1 = self.critic1(states, actions, z)
        value2 = self.critic2(states, actions, z)
        return value1, value2, {"z": z}


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
            self.models.append(self.create_model(fc_dim, sf_dim))

    def create_model(self, fc_dim, sf_dim):
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
