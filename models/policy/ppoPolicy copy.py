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
from models.layers.ppo_networks import PPO_Policy, PPO_Critic
from models.policy.base_policy import BasePolicy


class PPO_Learner(BasePolicy):
    def __init__(
        self,
        policy: PPO_Policy,
        critic: PPO_Critic,
        policy_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        eps: float = 0.2,
        entropy_scaler: float = 1e-2,
        gamma: float = 0.99,
        tau: float = 0.9,
        K: int = 5,
        device: str = "cpu",
    ):
        super(PPO_Learner, self).__init__()

        # constants
        self.device = device

        self._a_dim = policy._a_dim
        self._entropy_scaler = entropy_scaler
        self._eps = eps
        self._gamma = gamma
        self._tau = tau
        self._K = K
        self._forward_steps = 0

        # trainable networks
        self.policy = policy
        self.critic = critic

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.policy.parameters(), "lr": policy_lr},
                {"params": self.critic.parameters(), "lr": critic_lr},
            ]
        )

        #
        self.dummy = torch.tensor(0.0)
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    # def getPhi(self, x):
    #     with torch.no_grad():
    #         phi, _ = self.convNet(x)
    #     return phi

    def forward(self, obs, z=None, deterministic=False):
        self._forward_steps += 1
        x = obs["observation"]
        agent_pos = obs["agent_pos"]

        """
        x is state ~ (7, 7, 3)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x[None, :, :, :]
        x = x.to(self._dtype).to(self.device)
        raw_state = x.reshape(x.shape[0], -1)
        # phi = self.getPhi(x)

        # dist = self.policy(phi)
        dist = self.policy(raw_state)

        if deterministic:
            # [N, |O|]
            a = torch.argmax(dist.probs, dim=-1).to(self._dtype)
        else:
            a = dist.sample()

        logprobs = dist.log_prob(a)
        probs = torch.exp(logprobs)

        return a, {
            "q": self.dummy,  # dummy
            "phi": self.dummy,
            "is_option": False,  # dummy
            "z": self.dummy.item(),
            "probs": probs,
            "logprobs": logprobs,
            "entropy": dist.entropy(),
        }

    def learn(self, batch, z=0):
        self.train()
        t0 = time.time()

        # Ingredients
        (
            states,
            _,
            actions,
            _,
            _,
            _,
            _,
            rewards,
            terminals,
            old_logprobs,
        ) = self.preprocess_batch(batch, self.device)

        raw_states = states.reshape(states.shape[0], -1)

        with torch.no_grad():
            # values = self.critic(features)
            values = self.critic(raw_states)
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
            # critic ingredients
            # values = self.critic(features)
            values = self.critic(raw_states)
            valueLoss = F.mse_loss(returns, values)

            # policy ingredients
            # dist = self.policy(features)
            dist = self.policy(raw_states)

            logprobs = dist.log_prob(actions.squeeze()).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)

            ratios = torch.exp(logprobs - old_logprobs)
            # policy loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self._eps, 1 + self._eps) * advantages
            actorLoss = -torch.min(surr1, surr2)
            entropyLoss = self._entropy_scaler * entropy

            loss = torch.mean(actorLoss + 0.5 * valueLoss - entropyLoss)

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=20.0)
            grad_dict = self.compute_gradient_norm(
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
            (self.policy, self.critic),
            open(path, "wb"),
        )
        self.policy = self.policy.to(self.device)
        self.critic = self.critic.to(self.device)
