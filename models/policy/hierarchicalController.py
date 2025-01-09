import time
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import fmin_l_bfgs_b as bfgs

from copy import deepcopy
from utils.torch import get_flat_grad_from, get_flat_params_from, set_flat_params_to
from utils.utils import estimate_advantages
from models.layers.building_blocks import MLP
from models.policy.optionPolicy import OP_Controller
from models.layers.hc_networks import HC_Policy, HC_PPO, HC_Critic
from models.policy.base_policy import BasePolicy
from utils.normalizer import ObservationNormalizer


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
        sf_network: BasePolicy,
        op_network: OP_Controller,
        policy: HC_Policy,
        primitivePolicy: HC_PPO,
        critic: HC_Critic,
        minibatch_size: int,
        normalizer: ObservationNormalizer,
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
        self.minibatch_size = minibatch_size

        self._num_options = policy._num_options
        self._entropy_scaler = entropy_scaler
        self._a_dim = a_dim
        self._eps = eps
        self._gamma = gamma
        self._tau = tau
        self._K = K
        self._l2_reg = 1e-6
        self._target_kl = 0.03
        self._forward_steps = 0

        self.normalizer = normalizer

        # trainable networks
        self.policy = policy
        self.primitivePolicy = primitivePolicy
        self.critic = critic

        self.sf_network = sf_network
        self.op_network = op_network

        self.policy_optimizer = torch.optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": policy_lr},
                {"params": self.primitivePolicy.parameters(), "lr": policy_lr},
            ]
        )

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        #
        self.dummy = torch.tensor(1e-10)
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.sf_network.device = device
        self.op_network.device = device
        self.to(device)

    def getPhi(self, state, agent_pos):
        obs = {"observation": state, "agent_pos": agent_pos}
        with torch.no_grad():
            phi, _ = self.sf_network.get_features(obs, deterministic=True)
        return phi

    def normalize(self, obs):
        observation = obs["observation"]
        if self.normalizer is not None:
            observation = self.normalizer.normalize(observation)
        obs["observation"] = observation
        return obs

    def preprocess_obs(self, obs):
        observation = obs["observation"]
        agent_pos = obs["agent_pos"]

        if not torch.is_tensor(observation):
            observation = torch.from_numpy(observation).to(self._dtype).to(self.device)
        if agent_pos is not None and not torch.is_tensor(agent_pos):
            agent_pos = torch.from_numpy(agent_pos).to(self._dtype).to(self.device)

        return {"observation": observation, "agent_pos": agent_pos}

    def forward(self, obs, idx=None, deterministic=False):
        """
        Image-based state dimension ~ [Batch, width, height, channel] or [width, height, channel]
        Flat tensor-based state dimension ~ [Batch, tensor] or [tensor]
        """
        self._forward_steps += 1

        normalized_obs = self.preprocess_obs(obs)

        if idx is None:

            # sample a from the Hierarchical Policy
            z, z_argmax, metaData = self.policy(
                normalized_obs["observation"], deterministic=deterministic
            )
        else:
            # keep using the given z
            z = F.one_hot(idx, num_classes=self.policy._a_dim)
            z_argmax = idx
            probs = torch.tensor(1.0)
            metaData = {
                "probs": probs,
                "logprobs": torch.log(probs),
                "entropy": self.dummy,
            }  # dummy

        # print(z_argmax, self._num_options)
        is_option = True if z_argmax < self._num_options else False

        if is_option:
            # option selection
            # obs should be unnormalized
            with torch.no_grad():
                a, _ = self.op_network(obs, z_argmax, deterministic=deterministic)
        else:
            # primitive action selection
            a, _ = self.primitivePolicy(
                normalized_obs["observation"], deterministic=deterministic
            )

        return a, {
            "z": z,
            "z_argmax": z_argmax,
            "is_option": is_option,
            "is_hc_controller": True,
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
        }

    def learn(self, batch, prefix="HC"):
        self.train()
        t0 = time.time()

        # normalization
        if self.normalizer is not None:
            batch["states"] = self.normalizer.normalize(batch["states"], update=False)

        # Ingredients
        states = torch.from_numpy(batch["states"]).to(self._dtype).to(self.device)
        states = states.reshape(states.shape[0], -1)
        actions = torch.from_numpy(batch["actions"]).to(self._dtype).to(self.device)
        option_actions = (
            torch.from_numpy(batch["option_actions"]).to(self._dtype).to(self.device)
        )
        rewards = torch.from_numpy(batch["rewards"]).to(self._dtype).to(self.device)
        terminals = torch.from_numpy(batch["terminals"]).to(self._dtype).to(self.device)
        old_logprobs = (
            torch.from_numpy(batch["logprobs"]).to(self._dtype).to(self.device)
        )

        # Compute Advantage and returns of the current batch
        with torch.no_grad():
            values, _ = self.critic(states)
            _, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self._gamma,
                tau=self._tau,
                device=self.device,
            )

        # Minibatch setup
        batch_size = states.size(0)

        clip_fractions = []  # List to track clip fractions during training
        target_kl = []  # List to track target KL divergence values for early stopping

        policy_losses = []  # List to track policy loss over minibatches
        value_losses = []  # List to track value loss over minibatches
        entropy_losses = []

        actor_grads = []
        critic_grads = []

        # K - Loop
        for _ in range(self._K):
            indices = torch.randperm(batch_size)[: self.minibatch_size]
            mb_states, mb_actions = states[indices], actions[indices]
            mb_rewards, mb_terminals = rewards[indices], terminals[indices]
            mb_old_logprobs, mb_returns = old_logprobs[indices], returns[indices]
            mb_values, mb_option_actions = values[indices], option_actions[indices]

            # Recompute advantages for the minibatch
            with torch.no_grad():
                mb_advantages, _ = estimate_advantages(
                    mb_rewards,
                    mb_terminals,
                    mb_values,
                    gamma=self._gamma,
                    tau=self._tau,
                    device=self.device,
                )

            mb_values, _ = self.critic(mb_states)
            valueLoss = self.mse_loss(mb_values, mb_returns)

            # Add optional critic regularization
            l2_reg = (
                sum(param.pow(2).sum() for param in self.critic.parameters())
                * self._l2_reg
            )
            valueLoss += l2_reg

            # Update critic parameters
            self.critic_optimizer.zero_grad()
            valueLoss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            critic_grad_dict = self.compute_gradient_norm(
                [self.critic],
                ["critic"],
                dir="PPO",
                device=self.device,
            )
            critic_grads.append(critic_grad_dict)
            self.critic_optimizer.step()

            # Track value loss for logging
            value_losses.append(valueLoss.item())

            # find mask: the actions contributions by hc policy only
            hc_mask = torch.argmax(mb_option_actions, dim=-1) < self._num_options
            pm_mask = ~hc_mask

            # 2. Policy Update
            _, _, hc_metaData = self.policy(mb_states)
            _, pm_metaData = self.primitivePolicy(mb_states)

            # Compute hierarchical policy logprobs and entropy
            hc_logprobs = self.policy.log_prob(hc_metaData["dist"], mb_option_actions)[
                hc_mask
            ]
            hc_entropy = self.policy.entropy(hc_metaData["dist"])[hc_mask]

            pm_logprobs = self.primitivePolicy.log_prob(
                pm_metaData["dist"], mb_actions
            )[pm_mask]

            pm_entropy = self.primitivePolicy.entropy(pm_metaData["dist"])[pm_mask]

            # Combine logprobs and entropy in the original order
            logprobs = torch.empty_like(mb_returns)
            entropy = torch.empty_like(mb_returns)
            logprobs[hc_mask] = hc_logprobs
            logprobs[pm_mask] = pm_logprobs
            entropy[hc_mask] = hc_entropy
            entropy[pm_mask] = pm_entropy

            ratios = torch.exp(logprobs - mb_old_logprobs)

            surr1 = ratios * mb_advantages
            surr2 = torch.clamp(ratios, 1 - self._eps, 1 + self._eps) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.entropy_scaler * entropy.mean()
            policy_loss = actor_loss + entropy_loss

            # Track policy loss for logging
            policy_losses.append(policy_loss.item())
            entropy_losses.append(entropy_loss.item())

            # Compute clip fraction (for logging)
            clip_fraction = torch.mean(
                (torch.abs(ratios - 1) > self._eps).float()
            ).item()
            clip_fractions.append(clip_fraction)

            # Check if KL divergence exceeds target KL for early stopping
            kl_div = torch.mean(mb_old_logprobs - logprobs)
            target_kl.append(kl_div.item())
            if kl_div.item() > self.target_kl:
                print(f"Early stopping due to target KL divergence: {kl_div.item()}")
                break

            # Update policy parameters
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(
                self.primitivePolicy.parameters(), max_norm=0.5
            )
            actor_grad_dict = self.compute_gradient_norm(
                [self.policy, self.primitivePolicy],
                ["policy", "primitivePolicy"],
                dir="PPO",
                device=self.device,
            )
            actor_grads.append(actor_grad_dict)
            self.policy_optimizer.step()

        norm_dict = self.compute_weight_norm(
            [self.policy, self.primitivePolicy, self.critic],
            ["policy", "primitivePolicy", "critic"],
            dir=prefix,
            device=self.device,
        )

        loss_dict = {
            f"{prefix}/actorLoss": np.mean(policy_losses),
            f"{prefix}/valueLoss": np.mean(value_losses),
            f"{prefix}/entropyLoss": np.mean(entropy_losses),
            f"{prefix}/avg_reward": rewards.mean().item(),
            f"{prefix}/clip_fraction": np.mean(clip_fractions),
            f"{prefix}/target_kl": np.mean(target_kl),
        }
        actor_grad_norm = self.average_dict_values(actor_grads)
        critic_grad_norm = self.average_dict_values(critic_grads)

        loss_dict.update(actor_grad_norm)
        loss_dict.update(critic_grad_norm)
        loss_dict.update(norm_dict)

        del states, actions, option_actions, rewards, terminals, old_logprobs
        torch.cuda.empty_cache()

        t1 = time.time()
        self.eval()
        return (
            loss_dict,
            t1 - t0,
        )

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values and counts for each key
        sum_dict = {}
        count_dict = {}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                if key not in sum_dict:
                    sum_dict[key] = 0
                    count_dict[key] = 0
                sum_dict[key] += value
                count_dict[key] += 1

        # Calculate the average for each key
        avg_dict = {key: sum_val / count_dict[key] for key, sum_val in sum_dict.items()}

        return avg_dict

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
            (self.policy, self.primitivePolicy, self.critic, self.normalizer),
            open(path, "wb"),
        )
        self.policy = self.policy.to(self.device)
        self.primitivePolicy = self.primitivePolicy.to(self.device)
        self.critic = self.critic.to(self.device)
