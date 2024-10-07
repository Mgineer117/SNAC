import time
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from utils.utils import estimate_advantages
from models.layers.building_blocks import MLP
from models.layers.sf_networks import ConvNetwork, PsiCritic
from models.policy.optionPolicy import OP_Controller
from models.layers.hc_networks import HC_Policy, HC_PrimitivePolicy, HC_Critic
from models.policy.base_policy import BasePolicy


def check_all_devices(module):
    devices = {param.device for param in module.parameters()}  # Get all unique devices
    return devices


def compare_network_weights(model1: nn.Module, model2: nn.Module) -> float:
    """
    Compare the weights of two models and return the mean squared error between them.

    Args:
        model1 (nn.Module): The first model to compare.
        model2 (nn.Module): The second model to compare.

    Returns:
        float: The mean squared error between the weights of the two models.
    """
    mse_loss = nn.MSELoss()
    total_mse = 0.0
    num_params = 0

    # Iterate through parameters of both models
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if param1.shape != param2.shape:
            raise ValueError(
                "Model parameters have different shapes, models might have different architectures."
            )

        # Calculate MSE between parameters
        mse = mse_loss(param1, param2)
        total_mse += mse.item()
        num_params += 1

    # Average MSE across all parameters
    average_mse = total_mse / num_params if num_params > 0 else 0.0
    print(average_mse)

    return average_mse


class HC_Controller(BasePolicy):
    def __init__(
        self,
        policy: HC_Policy,
        primitivePolicy: HC_PrimitivePolicy,
        critic: HC_Critic,
        convNet: ConvNetwork,
        optionPolicy: OP_Controller,
        a_dim: int,
        policy_lr: float = 5e-4,
        critic_lr: float = 1e-4,
        eps: float = 0.2,
        entropy_scaler: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.95,
        K: int = 5,
        device: str = "cpu",
    ):
        super(HC_Controller, self).__init__()
        # constants
        self.device = device

        self._num_options = policy._num_options - 1
        self._entropy_scaler = entropy_scaler
        self._a_dim = a_dim
        self._eps = eps
        self._gamma = gamma
        self._tau = tau
        self._K = K
        self._forward_steps = 0

        # trainable networks
        self.policy = policy
        self.primitivePolicy = primitivePolicy
        self.critic = critic
        self.optionPolicy = optionPolicy
        self.convNet = convNet

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.policy.parameters(), "lr": policy_lr},
                {"params": self.critic.parameters(), "lr": critic_lr},
            ]
        )

        #
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.optionPolicy.device = device
        self.to(device)

    def getPhi(self, x):
        with torch.no_grad():
            phi, _ = self.convNet(x)
        return phi

    def forward(self, x, idx=None, deterministic=False):
        self._forward_steps += 1
        """
        x is state ~ (7, 7, 3)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x[None, :, :, :]
        x = x.to(self._dtype).to(self.device)

        # if not self._is_randomWalk:
        phi = self.getPhi(x)
        # x = x.view(x.size(0), -1)

        if idx is None:
            z, metaData = self.policy(phi, deterministic)
        else:
            z = idx
            metaData = {"probs": None, "logprobs": None}  # dummy

        is_option = True if z < self._num_options else False
        # print(z, is_option, self._num_options)

        if is_option:
            # option selection
            a, option_metaData = self.optionPolicy(x, z, deterministic=deterministic)
            termination = option_metaData["termination"]
        else:
            # primitive action selection
            # a, _ = self.primitivePolicy(phi)
            # I selected randomized action for primitive to prevent
            # it dominates over option

            q = torch.rand((1, self._a_dim)).to(self.device)
            a = torch.argmax(q, dim=-1)

            termination = True
        # print(z_idx, is_option, actions.shape, a)

        return a, {
            "z": z,
            "phi": phi,
            "is_option": is_option,
            "termination": termination,
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
        }

    def learn(self, batch, prefix="HC"):
        self.train()
        t0 = time.time()

        # Ingredients
        features = torch.from_numpy(batch["features"]).to(self._dtype).to(self.device)
        actions = (
            torch.from_numpy(batch["option_actions"]).to(self._dtype).to(self.device)
        )
        rewards = torch.from_numpy(batch["rewards"]).to(self._dtype).to(self.device)
        terminals = torch.from_numpy(batch["terminals"]).to(torch.int32).to(self.device)
        old_logprobs = (
            torch.from_numpy(batch["logprobs"]).to(self._dtype).to(self.device)
        )

        with torch.no_grad():
            # values, _ = self.critic(states)
            values, _ = self.critic(features)

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
            values, _ = self.critic(features)
            valueLoss = F.mse_loss(returns, values)

            _, metaData = self.policy(features)
            dist = metaData["dist"]

            logprobs = dist.log_prob(actions.squeeze()).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self._eps, 1 + self._eps) * advantages

            actorLoss = -torch.min(surr1, surr2)
            entropyLoss = self._entropy_scaler * entropy

            loss = torch.mean(actorLoss + 0.5 * valueLoss + entropyLoss)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            grad_dict = self.compute_gradient_norm(
                [self.policy, self.primitivePolicy, self.critic],
                ["policy", "primitivePolicy", "critic"],
                dir=prefix,
                device=self.device,
            )
            self.optimizer.step()

        norm_dict = self.compute_weight_norm(
            [self.policy, self.primitivePolicy, self.critic],
            ["policy", "primitivePolicy", "critic"],
            dir=prefix,
            device=self.device,
        )

        loss_dict = {
            f"{prefix}/loss": loss.item(),
            f"{prefix}/actorLoss": torch.mean(actorLoss).item(),
            f"{prefix}/valueLoss": torch.mean(valueLoss).item(),
            f"{prefix}/entropyLoss": torch.mean(entropyLoss).item(),
            f"{prefix}/trainReturn": torch.mean(rewards).item(),
        }
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        t1 = time.time()
        self.eval()
        return (
            loss_dict,
            t1 - t0,
        )

    def save_model(
        self, logdir: str, epoch: int = None, name: str = None, is_best=False
    ):
        self.policy = self.policy.cpu()
        self.primitivePolicy = self.primitivePolicy.cpu()
        self.critic = self.critic.cpu()

        # save checkpoint
        if is_best:
            path = os.path.join(logdir, f"best_model_{name}.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")
        pickle.dump(
            (self.policy, self.primitivePolicy, self.critic),
            open(path, "wb"),
        )
        self.policy = self.policy.to(self.device)
        self.primitivePolicy = self.primitivePolicy.to(self.device)
        self.critic = self.critic.to(self.device)
