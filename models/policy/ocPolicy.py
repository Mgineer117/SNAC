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
from models.layers.sf_networks import ConvNetwork, PsiCritic
from models.layers.oc_networks import OC_Policy, OC_Critic
from models.policy.base_policy import BasePolicy


class OC_Learner(BasePolicy):
    def __init__(
        self,
        sf_network: BasePolicy,
        policy: OC_Policy,
        critic: OC_Critic,
        policy_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        eps: float = 0.2,
        entropy_scaler: float = 1e-3,
        termination_reg: float = 1e-2,
        gamma: float = 0.99,
        tau: float = 0.9,
        K: int = 5,
        device: str = "cpu",
    ):
        super(OC_Learner, self).__init__()

        # constants
        self.device = device

        self._a_dim = policy._a_dim
        self._entropy_scaler = entropy_scaler
        self._termination_reg = termination_reg
        self._eps = eps
        self._gamma = gamma
        self._tau = tau
        self._K = K
        self._l2_reg = 1e-6
        self._bfgs_iter = K
        self._forward_steps = 0

        # trainable networks
        self.sf_network = sf_network
        self.policy = policy
        self.critic = critic
        self.target_critic = deepcopy(self.critic)

        if critic_lr is None:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
            self.is_bfgs = True
        else:
            self.optimizer = torch.optim.Adam(
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

        # deterministic is eval signal in this code
        epsilon = self.policy.epsilon(is_eval=deterministic)

        # the first iteration where z is not given
        if z is None:
            greedy_option = self.critic.greedy_option(obs["observation"])
        else:
            greedy_option = z

        current_option = (
            np.random.choice(self.policy._num_options)
            if np.random.rand() < epsilon
            else greedy_option
        )

        a, metaData = self.policy(
            obs["observation"], z=current_option, deterministic=deterministic
        )

        option_termination = self.policy.predict_option_termination(
            obs["observation"], z=current_option
        )

        return a, {
            # "z": self.dummy.item(),
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "option_termination": option_termination,
        }

    def actor_loss(self, batch):
        """Train policy and terminations

        Args:
            states (_type_): _description_
        """
        states, next_states, option_actions, entropy, logprobs, rewards, terminals = (
            batch
        )

        index_matrix = F.one_hot(option_actions, num_classes=self.policy._num_options)
        option_term_prob = self.policy.get_terminations(states) @ index_matrix

        next_option_term_prob = self.policy.get_terminations(next_states) @ index_matrix
        next_option_term_prob = next_option_term_prob.detach()

        Q = self.critic(states).detach().squeeze()
        next_Q_prime = self.target_critic(next_states).detach().squeeze()

        # Target update gt
        gt = rewards + (1 - terminals) * self._gamma * (
            (1 - next_option_term_prob) * next_Q_prime @ index_matrix
            + next_option_term_prob * next_Q_prime.max(dim=-1)[0]
        )

        # The termination loss
        termination_loss = torch.mean(
            option_term_prob
            * (
                (Q @ index_matrix).detach()
                - Q.max(dim=-1)[0].detach()
                + self._termination_reg
            )
            * (1 - terminals)
        )

        # actor-critic policy gradient with entropy regularization
        entropy_loss = self._entropy_scaler * entropy

        policy_loss = -logprobs * (gt.detach() - Q @ index_matrix) - entropy_loss

        actor_loss = torch.mean(termination_loss + policy_loss)

        return actor_loss, {
            "termination_loss": termination_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
        }

    def critic_loss(self, batch):
        """
        Training Q
        """
        states, next_states, option_actions, _, _, rewards, terminals = batch

        Q = self.critic(states)
        next_Q_prime = self.target_critic(next_states).detach()

        index_matrix = F.one_hot(option_actions, num_classes=self.policy._num_options)

        next_option_term_prob = self.policy.get_terminations(next_states) @ index_matrix
        next_option_term_prob = next_option_term_prob.detach()

        # Target update gt
        gt = rewards + (1 - terminals) * self._gamma * (
            (1 - next_option_term_prob) * next_Q_prime @ index_matrix
            + next_option_term_prob * next_Q_prime.max(dim=-1)[0]
        )

        # to update Q we want to use the actual network, not the prime
        td_err = (Q @ index_matrix - gt.detach()).pow(2).mul(0.5).mean()
        return td_err, {}

    def learn_policy(self, batch):
        """_summary_

        Args:
            batch (_type_): Online batch from sampler
        """
        t0 = time.time()

        self.optimizer.zero_grad()
        actorLoss, metaData = self.actor_loss(batch)
        actorLoss.backward()
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
            "OC/actorLoss": actorLoss.item(),
            "OC/terminationLoss": torch.mean(metaData["termination_loss"]).item(),
            "OC/policyLoss": torch.mean(metaData["policy_loss"]).item(),
            "OC/entropyLoss": torch.mean(metaData["entropy_loss"]).item(),
            "OC/trainAvgReward": torch.mean(batch["rewards"]).item(),
        }
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        t1 = time.time()
        self.eval()
        return (
            loss_dict,
            t1 - t0,
        )

    def learn_critic(self, batch):
        """_summary_

        Args:
            batch (_type_): offline batch but live
        """
        t0 = time.time()
        if self.is_bfgs:
            # L-BFGS-F value network update
            def closure(flat_params):
                set_flat_params_to(self.policy, torch.tensor(flat_params))
                for param in self.critic.parameters():
                    if param.grad is not None:
                        param.grad.data.fill_(0)
                valueLoss, _ = self.critic_loss(batch)
                for param in self.critic.parameters():
                    valueLoss += param.pow(2).sum() * self._l2_reg
                valueLoss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)

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
        else:
            batch = self.divide_into_subbatches(batch, self._bfgs_iter)
            for minibatch in batch:
                self.optimizer.zero_grad()
                valueLoss, _ = self.critic_loss(minibatch)
                valueLoss.backward()
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

        # migrate the parameters
        self.target_critic.load_state_dict(self.critic.state_dict())

        loss_dict = {
            "OC/valueLoss": valueLoss.item(),
        }
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        t1 = time.time()
        self.eval()
        return (
            loss_dict,
            t1 - t0,
        )

    def divide_into_subbatches(batch, subbatch_size):
        """
        Divide a batch of dictionaries into sub-batches.

        Args:
            batch (dict): A dictionary where each value is a list or tensor of equal length.
            subbatch_size (int): The size of each sub-batch.

        Returns:
            List[dict]: A list of dictionaries representing sub-batches.
        """
        keys = batch.keys()
        num_samples = len(next(iter(batch.values())))  # Get the size of the batch
        subbatches = []

        for i in range(0, num_samples, subbatch_size):
            subbatch = {
                key: value[i : i + subbatch_size] for key, value in batch.items()
            }
            subbatches.append(subbatch)

        return subbatches

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
