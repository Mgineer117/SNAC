import time
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import math
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import fmin_l_bfgs_b as bfgs

from utils.torch import get_flat_grad_from, get_flat_params_from, set_flat_params_to
from utils.utils import estimate_advantages, estimate_psi
from models.layers.building_blocks import MLP
from utils.normalizer import ObservationNormalizer
from models.layers.op_networks import OptionPolicy, OP_Critic, PsiCritic2
from models.policy.base_policy import BasePolicy


def check_all_devices(module):
    devices = {param.device for param in module.parameters()}  # Get all unique devices
    return devices


class OP_Controller(BasePolicy):
    def __init__(
        self,
        sf_network: BasePolicy,
        optionPolicy: OptionPolicy,
        optionCritic: OP_Critic,
        alpha: torch.Tensor | None,
        normalizer: ObservationNormalizer,
        optimizers: dict,
        options: nn.Module,
        option_vals: torch.Tensor,
        is_bfgs: bool,
        use_psi_action: bool,
        args,
    ):
        super(OP_Controller, self).__init__()

        # constants
        self.device = args.device
        self.algo_name = args.algo_name
        self.options = nn.Parameter(options.to(self._dtype).to(self.device))
        self.option_vals = option_vals.to(self._dtype).to(self.device)
        self.num_options = options.shape[0]
        self.use_psi_action = use_psi_action
        self.is_bfgs = is_bfgs
        self.a_dim = args.a_dim
        self.entropy_scaler = args.op_entropy_scaler
        self.eps = args.eps_clip
        self.gamma = args.gamma
        self.tau = args.tau
        self.K = args.OP_K_epochs
        self.l2_reg = 1e-6
        self.bfgs_iter = args.bfgs_iter
        self.is_discrete = args.is_discrete

        self.normalizer = normalizer

        # some sac params
        self.tune_alpha = args.tune_alpha
        self.soft_update_rate = args.sac_soft_update_rate
        self.target_update_interval = args.target_update_interval
        self.target_entropy = -args.a_dim

        # trainable networks
        self.sf_network = sf_network
        self.optionPolicy = optionPolicy
        self.optionCritic = optionCritic
        if args.op_mode == "sac":
            if alpha is not None:
                self.alpha = torch.tensor(alpha, device=args.device)
            else:
                self.alpha = torch.full(
                    (args.num_vector,), args.sac_init_alpha, device=args.device
                )

            if self.tune_alpha:
                self.log_alpha = nn.Parameter(torch.log(self.alpha))
                optimizers["alpha"] = torch.optim.Adam(
                    [self.log_alpha], lr=args.sac_alpha_lr
                )

            self.targetOptionCritic = deepcopy(optionCritic)

            self.num_update = 1

        self.optimizers = optimizers

        # inherent variable
        self._forward_steps = 0

        #
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.sf_network.device = device
        self.to(device)

    def preprocess_obs(self, obs):
        observation = obs["observation"]
        agent_pos = obs["agent_pos"]

        if self.normalizer is not None:
            observation = self.normalizer.normalize(observation)

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

        if self.use_psi_action:
            with torch.no_grad():
                psi, _ = self.sf_network.get_cumulative_features(obs)
            psi_logits = self._intrinsicValue(psi, z)
            probs = F.softmax(psi_logits, dim=-1)
            logprobs = F.log_softmax(psi_logits, dim=-1)
            entropy = -torch.sum(probs * logprobs, dim=-1)

            a_argmax = torch.argmax(probs, dim=-1)
            a = F.one_hot(a_argmax.long(), num_classes=self.a_dim)
            probs = torch.argmax(probs, dim=-1)
            logprobs = torch.argmax(logprobs, dim=-1)

            metaData = {"probs": probs, "logprobs": logprobs, "entropy": entropy}
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
            a = torch.randint(0, self.a_dim, (1,))
            a = F.one_hot(a, num_classes=self.a_dim)
        else:
            a = torch.rand((self.a_dim,))
        return a, {}

    def _intrinsicValue(self, psi, z):
        option = self.options[z, :]

        if self.algo_name in ("SNAC", "SNAC+", "SNAC++", "SNAC+++"):
            # divide phi in half
            psi_r, psi_s = self.split(psi)
            if z < int(self.num_options / 2):
                psi = psi_r
            else:
                psi = psi_s

        value = self.multiply_options(psi, option).squeeze(-1)
        return value

    def _intricsicReward(self, phi, next_phi, z):
        option = self.options[z, :]

        if self.algo_name in ("SNAC", "SNAC+", "SNAC++", "SNAC+++"):
            # divide phi in half
            phi_r, phi_s = self.split(phi)
            next_phi_r, next_phi_s = self.split(next_phi)
            if z < int(self.num_options / 2):
                deltaPhi = next_phi_r - phi_r  # N x F/2
            else:
                deltaPhi = next_phi_s - phi_s  # N x F/2
        else:
            deltaPhi = next_phi - phi  # N x F
        rew = self.multiply_options(deltaPhi, option)
        return rew

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_update_rate)
                + param.data * self.soft_update_rate
            )

    def learn(self, batch, z, mode="sac"):
        if mode == "ppo":
            loss_dict, update_time = self.ppo_learn(batch, z)
        elif mode == "sac":
            loss_dict, update_time = self.sac_learn(batch, z)
        else:
            raise NotImplementedError(f"{mode} is not implemented")
        return loss_dict, update_time

    def sac_learn(self, batch, z):
        self.train()
        t0 = time.time()

        # normalization
        if self.normalizer is not None:
            batch["states"] = self.normalizer.normalize(batch["states"], update=False)
            batch["next_states"] = self.normalizer.normalize(
                batch["next_states"], update=False
            )

        # Ingredients
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
            next_actions, next_meta = self.optionPolicy(next_states, z)
            next_q1, next_q2, _ = self.optionCritic(next_states, next_actions, z)
            next_q = torch.min(next_q1, next_q2) - self.alpha[z] * next_meta["logprobs"]
            target_q = rewards + (1 - terminals) * self.gamma * next_q

        q1, q2, _ = self.optionCritic(states, actions, z)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

        self.optimizers["critic"].zero_grad()
        critic_loss.backward()
        grad_dict1 = self.compute_gradient_norm(
            [self.optionCritic],
            ["optionCritic"],
            dir="OP_SAC",
            device=self.device,
        )
        self.optimizers["critic"].step()

        # Policy Loss
        new_actions, new_meta = self.optionPolicy(states, z)
        q1_new, q2_new, _ = self.optionCritic(states, new_actions, z)
        q_new = torch.min(q1_new, q2_new)  # Ensure this is out-of-place
        policy_loss = (self.alpha[z] * new_meta["logprobs"] - q_new).mean()

        self.optimizers["policy"].zero_grad()
        policy_loss.backward()
        grad_dict2 = self.compute_gradient_norm(
            [self.optionPolicy],
            ["optionPolicy"],
            dir="OP_SAC",
            device=self.device,
        )
        self.optimizers["policy"].step()

        # Alpha Loss
        if self.tune_alpha:
            alpha_loss = -(
                (
                    self.log_alpha[z]
                    * (new_meta["logprobs"] + self.target_entropy).detach()
                ).mean()
            )

            self.optimizers["alpha"].zero_grad()
            alpha_loss.backward()
            self.optimizers["alpha"].step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)

        # Soft update of target networks
        if self.num_update % self.target_update_interval == 0:
            self.soft_update(self.targetOptionCritic, self.optionCritic)

        self.num_update += 1

        # Log losses
        loss_dict = {
            "OP_SAC/critic_loss": critic_loss.item(),
            "OP_SAC/policy_loss": policy_loss.item(),
            "OP_SAC/alpha_loss": alpha_loss.item(),
            f"OP_SAC/alpha: {z}": self.alpha[z].item(),
            f"OP_SAC/IntEpRew:{z}": (torch.sum(rewards) / torch.sum(terminals)).item(),
        }
        norm_dict = self.compute_weight_norm(
            [self.optionPolicy, self.optionCritic],
            ["policy", "critic"],
            dir="OP_SAC",
            device=self.device,
        )
        loss_dict.update(norm_dict)
        loss_dict.update(grad_dict1)
        loss_dict.update(grad_dict2)

        update_time = time.time() - t0
        return loss_dict, update_time

    def ppo_learn(self, batch, z):
        self.train()
        t0 = time.time()

        # normalization
        if self.normalizer is not None:
            batch["states"] = self.normalizer.normalize(batch["states"], update=False)
            batch["next_states"] = self.normalizer.normalize(
                batch["next_states"], update=False
            )

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

        # Minibatch setup
        batch_size = states.size(0)
        minibatch_size = 2 * (batch_size // self.K)

        # K - Loop
        for _ in range(self.K):
            indices = torch.randperm(batch_size)[:minibatch_size]
            mb_states = states[indices]
            mb_actions = actions[indices]
            mb_rewards = rewards[indices]
            mb_terminals = terminals[indices]
            mb_old_logprobs = old_logprobs[indices]

            # Compute Advantage and returns of the current batch
            with torch.no_grad():
                mb_values, _ = self.optionCritic(mb_states, z)
                mb_advantages, mb_returns = estimate_advantages(
                    mb_rewards,
                    mb_terminals,
                    mb_values,
                    gamma=self.gamma,
                    tau=self.tau,
                    device=self.device,
                )

            valueLoss = self.mse_loss(mb_returns, mb_values)

            if self.is_bfgs:
                # L-BFGS-F value network update
                def closure(flat_params):
                    set_flat_params_to(self.optionCritic, torch.tensor(flat_params))
                    for param in self.optionCritic.parameters():
                        if param.grad is not None:
                            param.grad.data.fill_(0)
                    mb_values, _ = self.optionCritic(mb_states, z)
                    valueLoss = self.mse_loss(mb_values, mb_returns)
                    for param in self.optionCritic.parameters():
                        valueLoss += param.pow(2).sum() * self.l2_reg
                    valueLoss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.optionCritic.parameters(), max_norm=0.5
                    )

                    return (
                        valueLoss.item(),
                        get_flat_grad_from(self.optionCritic.parameters())
                        .cpu()
                        .numpy(),
                    )

                flat_params, _, opt_info = bfgs(
                    closure,
                    get_flat_params_from(self.optionCritic).detach().cpu().numpy(),
                    maxiter=self.bfgs_iter,
                )
                set_flat_params_to(self.optionCritic, torch.tensor(flat_params))

            _, metaData = self.optionPolicy(mb_states, z)

            logprobs = self.optionPolicy.log_prob(metaData["dist"], mb_actions)
            entropy = self.optionPolicy.entropy(metaData["dist"])

            ratios = torch.exp(logprobs - mb_old_logprobs)

            surr1 = ratios * mb_advantages
            surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * mb_advantages

            actorLoss = -torch.min(surr1, surr2)
            entropyLoss = self.entropy_scaler * entropy

            loss = torch.mean(actorLoss + 0.5 * valueLoss - entropyLoss)

            self.optimizers["ppo"].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            grad_dict = self.compute_gradient_norm(
                [self.optionPolicy, self.optionCritic],
                ["optionPolicy", "optionCritic"],
                dir="OP_PPO",
                device=self.device,
            )
            self.optimizers["ppo"].step()

        norm_dict = self.compute_weight_norm(
            [self.optionPolicy, self.optionCritic],
            ["policy", "critic"],
            dir="OP_PPO",
            device=self.device,
        )

        loss_dict = {
            "OP_PPO/loss": loss.item(),
            "OP_PPO/actorLoss": torch.mean(actorLoss).item(),
            "OP_PPO/valueLoss": torch.mean(valueLoss).item(),
            "OP_PPO/entropyLoss": torch.mean(entropyLoss).item(),
            f"OP_PPO/IntEpRew:{z}": (torch.sum(rewards) / torch.sum(terminals)).item(),
        }
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        del (
            states,
            actions,
            agent_pos,
            next_agent_pos,
            next_states,
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

    def save_model(self, logdir, epoch=None, is_best=False, mode="sac"):
        self.optionPolicy = self.optionPolicy.cpu()
        self.optionCritic = self.optionCritic.cpu()
        option_vals = self.option_vals.clone().cpu()
        options = nn.Parameter(self.options.clone().cpu())

        # save checkpoint
        if is_best:
            path = os.path.join(logdir, "best_model.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")

        if mode == "sac":
            alpha = nn.Parameter(self.alpha.clone().cpu())
            pickle.dump(
                (
                    self.optionPolicy,
                    self.optionCritic,
                    option_vals,
                    options,
                    alpha,
                    self.normalizer,
                ),
                open(path, "wb"),
            )
        elif mode == "ppo":
            pickle.dump(
                (
                    self.optionPolicy,
                    self.optionCritic,
                    option_vals,
                    options,
                ),
                open(path, "wb"),
            )

        self.optionPolicy = self.optionPolicy.to(self.device)
        self.optionCritic = self.optionCritic.to(self.device)
