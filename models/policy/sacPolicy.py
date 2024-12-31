import torch
import torch.nn as nn
import os
import time
import pickle
from utils.torch import get_flat_params_from, set_flat_params_to
from copy import deepcopy
from models.policy.base_policy import BasePolicy
from models.layers.sac_networks import SAC_Policy, SAC_CriticTwin


class SAC_Learner(BasePolicy):
    def __init__(
        self,
        policy: SAC_Policy,
        critic_twin: SAC_CriticTwin,
        alpha: float = 0.2,
        policy_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        trj_per_iter: int = 10,
        target_update_interval: int = 1,
        device="cpu",
    ):
        super(SAC_Learner, self).__init__()

        # Constants
        self.device = device
        self.alpha = nn.Parameter(torch.tensor(alpha)).to(self.device)
        self.gamma = gamma
        self.tau = tau
        self.trj_per_iter = trj_per_iter
        self.target_update_interval = target_update_interval
        self.num_update = 1

        # Networks
        self.policy = policy
        self.critic_twin = critic_twin
        self.target_critic_twin = deepcopy(critic_twin)

        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=policy_lr
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic_twin.parameters(), lr=critic_lr
        )
        self.alpha_optimizer = torch.optim.AdamW([self.alpha], lr=alpha_lr)

        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def preprocess_obs(self, obs):
        observation = obs["observation"]
        observation = torch.from_numpy(observation).to(torch.float32).to(self.device)
        return {"observation": observation}

    def forward(self, obs, z=None, deterministic=False):
        obs = self.preprocess_obs(obs)
        action, metaData = self.policy(obs["observation"], deterministic=deterministic)
        return action, metaData

    def learn(self, batch):
        self.train()
        t0 = time.time()

        states = torch.from_numpy(batch["states"]).to(torch.float32).to(self.device)
        states = states.reshape(states.shape[0], -1)
        actions = torch.from_numpy(batch["actions"]).to(torch.float32).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(torch.float32).to(self.device)
        next_states = (
            torch.from_numpy(batch["next_states"]).to(torch.float32).to(self.device)
        )
        next_states = next_states.reshape(next_states.shape[0], -1)
        terminals = (
            torch.from_numpy(batch["terminals"]).to(torch.float32).to(self.device)
        )

        # Critic Loss
        with torch.no_grad():
            next_actions, next_meta = self.policy(next_states)
            next_q1, next_q2 = self.target_critic_twin(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_meta["logprobs"]
            target_q = rewards + (1 - terminals) * self.gamma * next_q

        q1, q2 = self.critic_twin(states, actions)

        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

        # Policy Loss
        new_actions, new_meta = self.policy(states)
        with torch.no_grad():
            q1_new, q2_new = self.critic_twin(states, new_actions)
            q_new = torch.min(q1_new, q2_new)  # Ensure this is out-of-place
        policy_loss = (self.alpha * new_meta["logprobs"] - q_new).mean()

        # Alpha Loss
        alpha_loss = -(
            (self.alpha * (new_meta["logprobs"] + self.policy._a_dim).detach()).mean()
        )

        # Update networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.policy_optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()
        policy_loss.backward()
        alpha_loss.backward()
        self.policy_optimizer.step()
        self.alpha_optimizer.step()

        # Soft update of target networks
        if self.num_update % self.target_update_interval == 0:
            self.num_update = 1
            self.soft_update(self.target_critic_twin, self.critic_twin)
        else:
            self.num_update += 1

        # Log losses
        loss_dict = {
            "SAC/critic_loss": critic_loss.item(),
            "SAC/policy_loss": policy_loss.item(),
            "SAC/alpha_loss": alpha_loss.item(),
            "SAC/alpha": self.alpha.item(),
        }
        update_time = time.time() - t0
        return loss_dict, update_time

    def save_model(self, logdir, epoch=None, is_best=False):
        self.policy = self.policy.cpu()
        self.critic_twin = self.critic_twin.cpu()
        alpha = nn.Parameter(self.alpha.clone().cpu())

        if is_best:
            path = os.path.join(logdir, "best_model.p")
        else:
            path = os.path.join(logdir, f"model_{epoch}.p")

        pickle.dump(
            (self.policy, self.critic_twin, alpha),
            open(path, "wb"),
        )

        self.policy = self.policy.to(self.device)
        self.critic_twin = self.critic_twin.to(self.device)
