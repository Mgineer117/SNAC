import time
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import fmin_l_bfgs_b as bfgs

from utils.torch import get_flat_grad_from, get_flat_params_from, set_flat_params_to
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
        sf_network: BasePolicy,
        optionPolicy: OptionPolicy,
        optionCritic: OptionCritic,
        algo_name: str,
        options: nn.Module,
        option_vals: torch.Tensor,
        use_psi_action: bool,
        a_dim: int,
        policy_lr: float = 1e-4,
        critic_lr: float | None = None,
        eps: float = 0.2,
        entropy_scaler: float = 1e-3,
        gamma: float = 0.9,
        tau: float = 0.95,
        K: int = 5,
        is_discrete: bool = False,
        device: str = "cpu",
    ):
        super(OP_Controller, self).__init__()

        # constants
        self.device = device
        self._algo_name = algo_name
        self._options = nn.Parameter(options.to(self._dtype).to(self.device))
        self._option_vals = option_vals.to(self._dtype).to(self.device)
        self._num_options = options.shape[0]
        self._use_psi_action = use_psi_action
        self._a_dim = a_dim
        self._entropy_scaler = entropy_scaler
        self._eps = eps
        self._gamma = gamma
        self._tau = tau
        self._K = K
        self._l2_reg = 1e-6
        self._bfgs_iter = K
        self._forward_steps = 0
        self.is_discrete = is_discrete

        # trainable networks
        self.sf_network = sf_network
        self.optionPolicy = optionPolicy
        self.optionCritic = optionCritic

        self.optimizers = {}

        if critic_lr is None:
            self.optimizers["ppo"] = torch.optim.Adam(
                self.optionPolicy.parameters(), lr=policy_lr
            )
            self.is_bfgs = True
        else:
            self.optimizers["ppo"] = torch.optim.Adam(
                [
                    {"params": self.optionPolicy.parameters(), "lr": policy_lr},
                    {"params": self.optionCritic.parameters(), "lr": critic_lr},
                ]
            )
            self.is_bfgs = False
        #
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.sf_network.device = device
        self.to(device)

    def preprocess_obs(self, obs):
        observation = obs["observation"]
        agent_pos = obs["agent_pos"]

        if not torch.is_tensor(observation):
            observation = torch.from_numpy(observation).to(self._dtype).to(self.device)
        if agent_pos is not None and not torch.is_tensor(agent_pos):
            agent_pos = torch.from_numpy(agent_pos).to(self._dtype).to(self.device)

        return {"observation": observation, "agent_pos": agent_pos}

    def forward(self, obs, z, deterministic=False):
        """
        Image-based state dimension ~ [Batch, width, height, channel] or [width, height, channel]
        Flat tensor-based state dimension ~ [Batch, tensor] or [tensor]
        """
        self._forward_steps += 1
        obs = self.preprocess_obs(obs)

        if self._use_psi_action:
            with torch.no_grad():
                psi, _ = self.sf_network.get_cumulative_features(obs)
            psi_logits = self._intrinsicValue(psi, z)
            probs = F.softmax(psi_logits, dim=-1)
            logprobs = F.log_softmax(psi_logits, dim=-1)

            a_argmax = torch.argmax(probs, dim=-1)
            a = F.one_hot(a_argmax.long(), num_classes=self._a_dim)
            probs = torch.argmax(probs, dim=-1)
            logprobs = torch.argmax(logprobs, dim=-1)
            metaData = {"probs": probs, "logprobs": logprobs}
        else:
            a, metaData = self.optionPolicy(
                obs["observation"], z=z, deterministic=deterministic
            )

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
        }

    def random_walk(self, obs):
        if self.is_discrete:
            a = torch.randint(0, self._a_dim, (1,))
            a = F.one_hot(a, num_classes=self._a_dim)
        else:
            a = torch.rand((self._a_dim,))
        return a, {}

    def _intrinsicValue(self, psi, z):
        option = self._options[z, :]

        if self._algo_name in ("SNAC", "SNAC+", "SNAC++"):
            # divide phi in half
            psi_r, psi_s = self.split(psi)
            if z < int(self._num_options / 2):
                psi = psi_r
            else:
                psi = psi_s

        value = self.multiply_options(psi, option).squeeze(-1)
        return value

    def _intricsicReward(self, phi, next_phi, z):
        option = self._options[z, :]

        if self._algo_name in ("SNAC", "SNAC+", "SNAC++"):
            # divide phi in half
            phi_r, phi_s = self.split(phi)
            next_phi_r, next_phi_s = self.split(next_phi)
            if z < int(self._num_options / 2):
                deltaPhi = next_phi_r - phi_r  # N x F/2
                # deltaPhi = phi_r  # - phi_r  # N x F/2
            else:
                deltaPhi = next_phi_s - phi_s  # N x F/2
                # deltaPhi = phi_s  # - phi_s  # N x F/2
        else:
            deltaPhi = next_phi - phi  # N x F
            # deltaPhi = phi  # - phi  # N x F
        rew = self.multiply_options(deltaPhi, option)
        return rew

    def learn(self, batch, z):
        self.train()
        t0 = time.time()

        # Ingredients
        states = torch.from_numpy(batch["states"]).to(self._dtype).to(self.device)
        agent_pos = torch.from_numpy(batch["agent_pos"]).to(self._dtype).to(self.device)
        next_states = (
            torch.from_numpy(batch["next_states"]).to(self._dtype).to(self.device)
        )
        next_agent_pos = (
            torch.from_numpy(batch["next_agent_pos"]).to(self._dtype).to(self.device)
        )
        actions = torch.from_numpy(batch["actions"]).to(self._dtype).to(self.device)
        terminals = torch.from_numpy(batch["terminals"]).to(self._dtype).to(self.device)
        old_logprobs = (
            torch.from_numpy(batch["logprobs"]).to(self._dtype).to(self.device)
        )

        obs = {"observation": states, "agent_pos": agent_pos}
        next_obs = {"observation": next_states, "agent_pos": next_agent_pos}

        phi, _ = self.sf_network.get_features(obs)
        next_phi, _ = self.sf_network.get_features(next_obs)
        rewards = self._intricsicReward(phi, next_phi, z)

        states = states.reshape(states.shape[0], -1)

        # Compute Advantage and returns of the current batch
        with torch.no_grad():
            values, _ = self.optionCritic(states, z)
            advantages, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self._gamma,
                tau=self._tau,
                device=self.device,
            )
            valueLoss = self.mse_loss(returns, values)

        if self.is_bfgs:
            # L-BFGS-F value network update
            def closure(flat_params):
                set_flat_params_to(self.optionCritic, torch.tensor(flat_params))
                for param in self.optionCritic.parameters():
                    if param.grad is not None:
                        param.grad.data.fill_(0)
                values, _ = self.optionCritic(states, z)
                valueLoss = self.mse_loss(values, returns)
                for param in self.optionCritic.parameters():
                    valueLoss += param.pow(2).sum() * self._l2_reg
                valueLoss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.optionCritic.parameters(), max_norm=1.0
                )

                return (
                    valueLoss.item(),
                    get_flat_grad_from(self.optionCritic.parameters()).cpu().numpy(),
                )

            flat_params, _, opt_info = bfgs(
                closure,
                get_flat_params_from(self.optionCritic).detach().cpu().numpy(),
                maxiter=self._bfgs_iter,
            )
            set_flat_params_to(self.optionCritic, torch.tensor(flat_params))

        # K - Loop
        for _ in range(self._K):
            if not self.is_bfgs:
                values, _ = self.optionCritic(states, z)
                valueLoss = self.mse_loss(returns, values)

            _, metaData = self.optionPolicy(states, z)

            logprobs = self.optionPolicy.log_prob(metaData["dist"], actions)
            entropy = self.optionPolicy.entropy(metaData["dist"])

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self._eps, 1 + self._eps) * advantages

            actorLoss = -torch.min(surr1, surr2)
            entropyLoss = self._entropy_scaler * entropy

            loss = torch.mean(actorLoss + 0.5 * valueLoss - entropyLoss)

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

        norm_dict = self.compute_weight_norm(
            [self.optionPolicy, self.optionCritic],
            ["policy", "critic"],
            dir="OP",
            device=self.device,
        )

        loss_dict = {
            "OP/loss": loss.item(),
            "OP/actorLoss": torch.mean(actorLoss).item(),
            "OP/valueLoss": torch.mean(valueLoss).item(),
            "OP/entropyLoss": torch.mean(entropyLoss).item(),
        }
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        avgRewDict = {
            f"OP/IntEpRew:{z}": (torch.sum(rewards) / torch.sum(terminals)).item(),
        }
        t1 = time.time()
        self.eval()
        return (
            loss_dict,
            avgRewDict,
            t1 - t0,
        )

    def save_model(self, logdir, epoch=None, is_best=False):
        self.optionPolicy = self.optionPolicy.cpu()
        self.optionCritic = self.optionCritic.cpu()
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
                self._option_vals,
                self._options,
            ),
            open(path, "wb"),
        )
        self.optionPolicy = self.optionPolicy.to(self.device)
        self.optionCritic = self.optionCritic.to(self.device)
        self._options = nn.Parameter(self._options.to(self.device))
        self._option_vals = self._option_vals.to(self.device)
