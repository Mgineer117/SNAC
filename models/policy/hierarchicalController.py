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
        normalizer: ObservationNormalizer,
        a_dim: int,
        policy_lr: float = 5e-4,
        critic_lr: float = 1e-4,
        eps: float = 0.2,
        entropy_scaler: float = 1e-3,
        bfgs_iter: int = 5,
        gamma: float = 0.99,
        tau: float = 0.95,
        K: int = 5,
        device: str = "cpu",
    ):
        super(HC_Controller, self).__init__()
        # constants
        self.device = device

        self._num_options = policy._num_options
        self._entropy_scaler = entropy_scaler
        self._a_dim = a_dim
        self._eps = eps
        self._gamma = gamma
        self._tau = tau
        self._K = K
        self._l2_reg = 1e-6
        self._bfgs_iter = bfgs_iter
        self._forward_steps = 0

        self.normalizer = normalizer

        # trainable networks
        self.policy = policy
        self.primitivePolicy = primitivePolicy
        self.critic = critic

        self.sf_network = sf_network
        self.op_network = op_network

        if critic_lr is None:
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": self.policy.parameters(), "lr": policy_lr},
                    {"params": self.primitivePolicy.parameters(), "lr": policy_lr},
                ]
            )
            self.is_bfgs = True
        else:
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": self.policy.parameters(), "lr": policy_lr},
                    {"params": self.primitivePolicy.parameters(), "lr": policy_lr},
                    {"params": self.critic.parameters(), "lr": critic_lr},
                ]
            )
            self.is_bfgs = False

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

        # Minibatch setup
        batch_size = states.size(0)
        minibatch_size = 2 * (batch_size // self._K)

        # K - Loop
        for _ in range(self._K):
            indices = torch.randperm(batch_size)[:minibatch_size]
            mb_states = states[indices]
            mb_actions = actions[indices]
            mb_option_actions = option_actions[indices]
            mb_rewards = rewards[indices]
            mb_terminals = terminals[indices]
            mb_old_logprobs = old_logprobs[indices]

            # Compute Advantage and returns of the current batch
            with torch.no_grad():
                mb_values, _ = self.critic(mb_states)
                mb_advantages, mb_returns = estimate_advantages(
                    mb_rewards,
                    mb_terminals,
                    mb_values,
                    gamma=self._gamma,
                    tau=self._tau,
                    device=self.device,
                )

            valueLoss = self.mse_loss(mb_returns, mb_values)

            if self.is_bfgs:
                # L-BFGS-F value network update
                def closure(flat_params):
                    set_flat_params_to(self.critic, torch.tensor(flat_params))
                    for param in self.critic.parameters():
                        if param.grad is not None:
                            param.grad.data.fill_(0)
                    mb_values, _ = self.critic(mb_states)
                    valueLoss = self.mse_loss(mb_values, mb_returns)
                    for param in self.critic.parameters():
                        valueLoss += param.pow(2).sum() * self._l2_reg
                    valueLoss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), max_norm=0.5
                    )

                    return (
                        valueLoss.item(),
                        get_flat_grad_from(self.critic.parameters()).cpu().numpy(),
                    )

                flat_params, _, opt_info = bfgs(
                    closure,
                    get_flat_params_from(self.critic).detach().cpu().numpy(),
                    maxiter=self._bfgs_iter,
                )
                set_flat_params_to(self.critic, torch.tensor(flat_params))

            # find mask: the actions contributions by hc policy only
            hc_mask = torch.argmax(mb_option_actions, dim=-1) < self._num_options
            pm_mask = ~hc_mask

            _, _, hc_metaData = self.policy(mb_states)
            _, pm_metaData = self.primitivePolicy(mb_states)

            # Compute hierarchical policy logprobs and entropy
            hc_logprobs = self.policy.log_prob(hc_metaData["dist"], mb_option_actions)[
                hc_mask
            ]
            hc_entropy = self.policy.entropy(hc_metaData["dist"])[hc_mask]

            # Compute primitive policy logprobs and entropy
            pm_logprobs = self.primitivePolicy.log_prob(
                pm_metaData["dist"], mb_actions
            )[pm_mask]

            pm_entropy = self.primitivePolicy.entropy(pm_metaData["dist"])[pm_mask]

            # Combine logprobs and entropy in the original order
            logprobs = torch.empty_like(mb_rewards)
            entropy = torch.empty_like(mb_rewards)
            logprobs[hc_mask] = hc_logprobs
            logprobs[pm_mask] = pm_logprobs
            entropy[hc_mask] = hc_entropy
            entropy[pm_mask] = pm_entropy

            ratios = torch.exp(logprobs - mb_old_logprobs)

            surr1 = ratios * mb_advantages
            surr2 = torch.clamp(ratios, 1 - self._eps, 1 + self._eps) * mb_advantages

            actorLoss = -torch.min(surr1, surr2)
            entropyLoss = self._entropy_scaler * entropy

            loss = torch.mean(actorLoss + 0.5 * valueLoss - entropyLoss)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
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

        del states, actions, option_actions, rewards, terminals, old_logprobs
        torch.cuda.empty_cache()

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
            (self.policy, self.primitivePolicy, self.critic, self.normalizer),
            open(path, "wb"),
        )
        self.policy = self.policy.to(self.device)
        self.primitivePolicy = self.primitivePolicy.to(self.device)
        self.critic = self.critic.to(self.device)
