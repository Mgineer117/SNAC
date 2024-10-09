import time
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from utils.utils import estimate_advantages, estimate_psi
from models.layers.building_blocks import MLP
from models.layers.sf_networks import ConvNetwork, PsiCritic
from models.layers.op_networks import OptionPolicy, OptionCritic, PsiCritic2
from models.policy.base_policy import BasePolicy


def check_all_devices(module):
    devices = {param.device for param in module.parameters()}  # Get all unique devices
    return devices


class OP_Controller(BasePolicy):
    def __init__(
        self,
        optionPolicy: OptionPolicy,
        optionCritic: OptionCritic,
        convNet: ConvNetwork,
        psiNet: PsiCritic2,
        algo_name: str,
        options: nn.Module,
        option_vals: torch.Tensor,
        a_dim: int,
        policy_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        psi_lr: float = 3e-4,
        eps: float = 0.2,
        entropy_scaler: float = 1e-3,
        gamma: float = 0.9,
        tau: float = 0.95,
        K: int = 5,
        device: str = "cpu",
    ):
        super(OP_Controller, self).__init__()

        # constants
        self.device = device
        self._algo_name = algo_name
        self._options = nn.Parameter(options.to(self._dtype).to(self.device))
        self._option_vals = option_vals.to(self._dtype).to(self.device)
        self._num_options = options.shape[0]
        self._a_dim = a_dim
        self._entropy_scaler = entropy_scaler
        self._eps = eps
        self._gamma = gamma
        self._tau = tau
        self._K = K
        self._forward_steps = 0

        # trainable networks
        self.optionPolicy = optionPolicy
        self.optionCritic = optionCritic
        self.convNet = convNet
        self.psiNet = psiNet

        self.optimizers = {}

        self.optimizers["ppo"] = torch.optim.AdamW(
            [
                {"params": self.optionPolicy.parameters(), "lr": policy_lr},
                {"params": self.optionCritic.parameters(), "lr": critic_lr},
            ]
        )
        self.optimizers["psiNet"] = torch.optim.AdamW(
            self.psiNet.parameters(), lr=psi_lr
        )
        # self._options = self._options
        #
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def getPhi(self, x):
        with torch.no_grad():
            phi, _ = self.convNet(x)
        return phi

    def forward(self, x, z, deterministic=False):
        self._forward_steps += 1
        """
        x is state ~ (7, 7, 3)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x[None, :, :, :]
        x = x.to(self._dtype).to(self.device)
        raw_state = x.reshape(x.shape[0], -1)
        phi = self.getPhi(x)

        # a, metaData = self.optionPolicy(phi, z, deterministic)
        a, metaData = self.optionPolicy(raw_state, z, deterministic)

        # compute q
        psi, _ = self.psiNet(phi, z)

        if self._algo_name == "SNAC":
            psi_r, psi_s = self.split(psi)

            if z < (self._num_options / 2):
                q = self.multiply_options(psi_r, self._options[z, :]).squeeze()
            else:
                q = self.multiply_options(psi_s, self._options[z, :]).squeeze()
        elif self._algo_name == "EigenOption" or self._algo_name == "CoveringOption":
            q = self.multiply_options(psi, self._options[z, :]).squeeze()
        else:
            raise ValueError(f"algo_name is unknown: {self._algo_name}")

        return a, {
            "q": q,
            "phi": phi,
            "is_option": False,  # dummy
            "z": 0,
            "termination": False,
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
        }

    def _intricsicReward(self, phi, next_phi, z):
        # F x 1
        option = self._options[z, :]

        if self._algo_name == "SNAC":
            # divide phi in half
            phi_r, phi_s = self.split(phi)
            next_phi_r, next_phi_s = self.split(next_phi)

            if z < int(self._num_options / 2):
                deltaPhi = next_phi_r - phi_r  # N x F/2
            else:
                deltaPhi = next_phi_s - phi_s  # N x F/2

        elif self._algo_name == "EigenOption" or self._algo_name == "CoveringOption":
            deltaPhi = next_phi - phi  # N x F

        # N x 1
        rew = self.multiply_options(deltaPhi, option)
        return rew

    def learn(self, batch, z):
        self.train()
        t0 = time.time()

        # Ingredients
        (
            states,
            features,
            actions,
            _,
            next_states,
            rewards,
            terminals,
            old_logprobs,
        ) = self.preprocess_batch(batch, self.device)

        raw_states = states.reshape(states.shape[0], -1)

        phi = features
        next_phi = self.getPhi(next_states)

        rewards = self._intricsicReward(phi, next_phi, z)

        ### LEARN PPO ###
        with torch.no_grad():
            # values, _ = self.optionCritic(phi, z)
            values, _ = self.optionCritic(raw_states, z)

            advantages, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self._gamma,
                tau=self._tau,
                device=self.device,
            )

        # K - Loop
        for _ in range(self._K):
            # values, _ = self.optionCritic(phi, z)
            values, _ = self.optionCritic(raw_states, z)
            valueLoss = F.mse_loss(returns, values)

            # _, metaData = self.optionPolicy(phi, z)
            _, metaData = self.optionPolicy(raw_states, z)
            dist = metaData["dist"]

            logprobs = dist.log_prob(actions.squeeze()).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self._eps, 1 + self._eps) * advantages

            actorLoss = -torch.min(surr1, surr2)
            entropyLoss = self._entropy_scaler * entropy

            loss = torch.mean(actorLoss + 0.5 * valueLoss + entropyLoss)

            self.optimizers["ppo"].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            grad_dict = self.compute_gradient_norm(
                [self.optionPolicy, self.optionCritic],
                ["optionPolicy", "optionCritic"],
                dir="OP",
                device=self.device,
            )
            self.optimizers["ppo"].step()

        loss_dict = {
            "OP/loss": loss.item(),
            "OP/actorLoss": torch.mean(actorLoss).item(),
            "OP/valueLoss": torch.mean(valueLoss).item(),
            "OP/entropyLoss": torch.mean(entropyLoss).item(),
            "OP/intrinsicAvgReward": (torch.mean(rewards) / rewards.shape[0]).item(),
        }
        loss_dict.update(grad_dict)

        t1 = time.time()
        self.eval()
        return (
            loss_dict,
            t1 - t0,
        )

    def learnPsi(self, batch, z):
        self.train()
        t0 = time.time()

        # Ingredients
        _, features, actions_oh, _, _, terminals, _ = self.preprocess_batch(
            batch, self.device
        )

        ### LEARN OPTION PSI ###
        phi = features
        psi, _ = self.psiNet(phi, z)
        filteredPsi = torch.sum(
            psi * actions_oh.unsqueeze(-1), axis=1
        )  # -> filteredPsi ~ [N, F] since no keepdim=True

        psi_est = estimate_psi(phi, terminals, self._gamma, self.device)
        psi_loss = self.huber_loss(psi_est, filteredPsi)

        self.optimizers["psiNet"].zero_grad()
        psi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        grad_dict = self.compute_gradient_norm(
            [self.psiNet],
            ["psiNet"],
            dir="OP",
            device=self.device,
        )
        self.optimizers["psiNet"].step()

        loss_dict = {
            "OP/psiLoss": psi_loss.item(),
        }
        loss_dict.update(grad_dict)

        t1 = time.time()
        self.eval()
        return (
            loss_dict,
            t1 - t0,
        )

    def save_model(self, logdir, epoch=None, is_best=False):
        self.optionPolicy = self.optionPolicy.cpu()
        self.optionCritic = self.optionCritic.cpu()
        self.psiNet = self.psiNet.cpu()
        self._options = nn.Parameter(self._options.cpu())
        self._option_vals = self._option_vals.cpu()

        # save checkpoint
        if is_best:
            path = os.path.join(logdir, "best_model.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")
        pickle.dump(
            (
                self.optionPolicy,
                self.optionCritic,
                self.psiNet,
                self._option_vals,
                self._options,
            ),
            open(path, "wb"),
        )
        self.optionPolicy = self.optionPolicy.to(self.device)
        self.optionCritic = self.optionCritic.to(self.device)
        self.psiNet = self.psiNet.to(self.device)
        self._options = nn.Parameter(self._options.to(self.device))
        self._option_vals = self._option_vals.to(self.device)
