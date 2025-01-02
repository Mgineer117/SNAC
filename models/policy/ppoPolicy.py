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
from utils import estimate_advantages
from models.layers.building_blocks import MLP
from utils.normalizer import ObservationNormalizer
from models.layers.ppo_networks import PPO_Policy, PPO_Critic
from models.policy.base_policy import BasePolicy


class PPO_Learner(BasePolicy):
    def __init__(
        self,
        policy: PPO_Policy,
        critic: PPO_Critic,
        normalizer: ObservationNormalizer,
        policy_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        eps: float = 0.2,
        entropy_scaler: float = 1e-3,
        bfgs_iter: int = 5,
        gamma: float = 0.99,
        tau: float = 0.9,
        K: int = 5,
        device: str = "cpu",
    ):
        super(PPO_Learner, self).__init__()

        # constants
        self.device = device

        self._entropy_scaler = entropy_scaler
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
        self.critic = critic

        if critic_lr is None:
            self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=policy_lr)
            self.is_bfgs = True
        else:
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": self.policy.parameters(), "lr": policy_lr},
                    {"params": self.critic.parameters(), "lr": critic_lr},
                ]
            )
            self.is_bfgs = False

        #
        self.dummy = torch.tensor(0.0)
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def preprocess_obs(self, obs):
        observation = obs["observation"]
        agent_pos = obs["agent_pos"]

        if self.normalizer is not None:
            observation = self.normalizer.normalize(observation)

        # preprocessing
        observation = torch.from_numpy(observation).to(self._dtype).to(self.device)
        if np.any(agent_pos != None):
            agent_pos = torch.from_numpy(agent_pos).to(self._dtype).to(self.device)

        return {"observation": observation, "agent_pos": agent_pos}

    def forward(self, obs, z=None, deterministic=False):
        """
        Image-based state dimension ~ [Batch, width, height, channel] or [width, height, channel]
        Flat tensor-based state dimension ~ [Batch, tensor] or [tensor]
        z is dummy input for code consistency
        """
        self._forward_steps += 1
        obs = self.preprocess_obs(obs)

        a, metaData = self.policy(obs["observation"], deterministic=deterministic)

        return a, {
            # "z": self.dummy.item(),
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
        }

    def learn(self, batch, z=0):
        self.train()
        t0 = time.time()

        # normalization
        if self.normalizer is not None:
            batch["states"] = self.normalizer.normalize(batch["states"], update=False)

        # Ingredients
        states = torch.from_numpy(batch["states"]).to(self._dtype).to(self.device)
        states = states.reshape(states.shape[0], -1)
        actions = torch.from_numpy(batch["actions"]).to(self._dtype).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self._dtype).to(self.device)
        terminals = torch.from_numpy(batch["terminals"]).to(self._dtype).to(self.device)
        old_logprobs = (
            torch.from_numpy(batch["logprobs"]).to(self._dtype).to(self.device)
        )

        # Minibatch setup
        batch_size = states.size(0)
        minibatch_size = 2 * (batch_size // self._K)

        # K - Loop with minibatch training
        for _ in range(self._K):
            indices = torch.randperm(batch_size)[:minibatch_size]
            mb_states = states[indices]
            mb_actions = actions[indices]
            mb_rewards = rewards[indices]
            mb_terminals = terminals[indices]
            mb_old_logprobs = old_logprobs[indices]

            # Compute Advantage and returns of the current batch
            with torch.no_grad():
                mb_values = self.critic(mb_states)
                mb_advantages, mb_returns = estimate_advantages(
                    mb_rewards,
                    mb_terminals,
                    mb_values,
                    gamma=self._gamma,
                    tau=self._tau,
                    device=self.device,
                )

            valueLoss = self.mse_loss(mb_returns, mb_values)

            # Update value function (critic)
            if self.is_bfgs:
                # L-BFGS-F value network update
                def closure(flat_params):
                    set_flat_params_to(self.critic, torch.tensor(flat_params))
                    for param in self.critic.parameters():
                        if param.grad is not None:
                            param.grad.data.fill_(0)
                    mb_values = self.critic(mb_states)
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

            # policy ingredients
            _, metaData = self.policy(mb_states)
            logprobs = self.policy.log_prob(metaData["dist"], mb_actions)
            entropy = self.policy.entropy(metaData["dist"])
            ratios = torch.exp(logprobs - mb_old_logprobs)

            # policy loss
            surr1 = ratios * mb_advantages
            surr2 = torch.clamp(ratios, 1 - self._eps, 1 + self._eps) * mb_advantages
            actorLoss = -torch.min(surr1, surr2)
            entropyLoss = self._entropy_scaler * entropy
            loss = torch.mean(actorLoss - entropyLoss)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            grad_dict = self.compute_gradient_norm(
                [self.policy, self.critic],
                ["policy", "critic"],
                dir="PPO",
                device=self.device,
            )
            norm_dict = self.compute_weight_norm(
                [self.policy, self.critic],
                ["policy", "critic"],
                dir="PPO",
                device=self.device,
            )
            self.optimizer.step()

        loss_dict = {
            "PPO/loss": loss.item(),
            "PPO/actorLoss": torch.mean(actorLoss).item(),
            "PPO/valueLoss": torch.mean(valueLoss).item(),
            "PPO/entropyLoss": torch.mean(entropyLoss).item(),
            "PPO/trainAvgReward": (torch.sum(rewards) / rewards.shape[0]).item(),
        }
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        del (
            states,
            actions,
            rewards,
            terminals,
            old_logprobs,
        )
        torch.cuda.empty_cache()

        t1 = time.time()
        self.eval()
        return (
            loss_dict,
            t1 - t0,
        )

    def save_model(self, logdir, epoch=None, is_best=False):
        self.policy = self.policy.cpu()
        self.critic = self.critic.cpu()

        # save checkpoint
        if is_best:
            path = os.path.join(logdir, "best_model.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")
        pickle.dump(
            (self.policy, self.critic, self.normalizer),
            open(path, "wb"),
        )
        self.policy = self.policy.to(self.device)
        self.critic = self.critic.to(self.device)
