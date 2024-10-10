import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os


class BasePolicy(nn.Module):
    def __init__(self):
        super(BasePolicy, self).__init__()

        # networks
        self.feaNet = None
        self.psiNet = None
        self.dcnNet = None
        self._options = None

        self.device = torch.device("cpu")

        # constants
        self._dtype = torch.float32

        # utils
        self.l1_loss = F.l1_loss
        self.mse_loss = F.mse_loss
        self.huber_loss = F.smooth_l1_loss
        self.mqe_loss = lambda x, y: torch.mean(
            torch.mean(
                torch.mean(torch.mean(torch.pow(x - y, 4), -1), axis=-1), axis=-1
            ),
            axis=0,
        )

        # self.multiply_options = lambda x, y: torch.einsum(
        #     "naf,nf->na", x, y
        # )  # ~ [N, |A|]
        self.multiply_options = lambda x, y: torch.sum(
            torch.mul(x, y), axis=-1, keepdim=True
        )

    def preprocess_batch(self, batch, device):
        (
            states,
            features,
            actions,
            next_states,
            agent_pos,
            next_agent_pos,
            rewards,
            terminals,
            logprobs,
        ) = (
            batch["states"],
            batch["features"],
            batch["actions"],
            batch["next_states"],
            batch["agent_pos"],
            batch["next_agent_pos"],
            batch["rewards"],
            batch["terminals"],
            batch["logprobs"],
        )

        states = torch.from_numpy(states).to(self._dtype).to(device)
        features = torch.from_numpy(features).to(self._dtype).to(device)
        actions = torch.from_numpy(actions).to(self._dtype).to(device)
        actions_oh = (
            F.one_hot(actions.long(), num_classes=self._a_dim).to(self._dtype).squeeze()
        )
        next_states = torch.from_numpy(next_states).to(self._dtype).to(device)
        agent_pos = torch.from_numpy(agent_pos).to(self._dtype).to(device)
        next_agent_pos = torch.from_numpy(next_agent_pos).to(self._dtype).to(device)
        rewards = torch.from_numpy(rewards).to(self._dtype).to(device)
        terminals = torch.from_numpy(terminals).to(torch.int32).to(device)
        logprobs = torch.from_numpy(logprobs).to(self._dtype).to(device)

        return (
            states,
            features,
            actions,
            actions_oh,
            next_states,
            agent_pos,
            next_agent_pos,
            rewards,
            terminals,
            logprobs,
        )

    def compute_gradient_norm(self, models, names, device, dir="None", norm_type=2):
        grad_dict = {}
        for i, model in enumerate(models):
            total_norm = torch.tensor(0.0, device=device)
            try:
                for param in model.parameters():
                    if (
                        param.grad is not None
                    ):  # Only consider parameters that have gradients
                        param_grad_norm = torch.norm(param.grad, p=norm_type)
                        total_norm += param_grad_norm**norm_type
            except:
                try:
                    param_grad_norm = torch.norm(model.grad, p=norm_type)
                except:
                    param_grad_norm = torch.tensor(0.0)
                total_norm += param_grad_norm**norm_type

            total_norm = total_norm ** (1.0 / norm_type)
            grad_dict[dir + "/grad/" + names[i]] = total_norm.item()

        return grad_dict

    def compute_weight_norm(self, models, names, device, dir="None", norm_type=2):
        norm_dict = {}
        for i, model in enumerate(models):
            total_norm = torch.tensor(0.0, device=device)
            try:
                for param in model.parameters():
                    param_norm = torch.norm(param, p=norm_type)
                    total_norm += param_norm**norm_type
            except:
                param_norm = torch.norm(model, p=norm_type)
                total_norm += param_norm**norm_type
            total_norm = total_norm ** (1.0 / norm_type)
            norm_dict[dir + "/weight/" + names[i]] = total_norm.item()

        return norm_dict

    def learn(self):
        pass

    def learnPsi(self):
        pass

    def discover_options(self):
        pass

    def split(self, x):
        x_r, x_s = torch.split(x, x.size(-1) // 2, dim=-1)
        return x_r, x_s
