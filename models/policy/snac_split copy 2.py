import time
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from models.layers.building_blocks import MLP
from models.layers.op_networks import ConvNetwork, DecisionNetwork, Critic
from models.policy.base_policy import BasePolicy


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


class SNAC_Split(BasePolicy):
    def __init__(
        self,
        convNet: ConvNetwork,
        dcnNet: DecisionNetwork,
        qNet: Critic,
        a_dim: int,
        options=None,
        update_iter: int = 5,
        trj_per_iter: int = 10,
        decision_mode: str = "random",
        gamma: float = 0.99,
        epsilon: float = 0.2,
        anneal: float = 1e-5,
        phi_loss_r_scaler: float = 1.0,
        phi_loss_s_scaler: float = 0.1,
        psi_loss_scaler: float = 1.0,
        q_loss_scaler: float = 0.0,
        device: str = "cpu",
    ):
        super(SNAC_Split, self).__init__()

        # constants
        self.decision_mode = decision_mode
        self.device = device

        self._update_iter = update_iter
        self._trj_per_iter = trj_per_iter
        self._a_dim = a_dim
        self._fc_dim = convNet._fc_dim
        self._sf_dim = convNet._sf_dim
        self._epsilon = epsilon
        self._gamma = gamma
        self._anneal = anneal
        self._forward_steps = 0
        self._phi_loss_r_scaler = phi_loss_r_scaler
        self._phi_loss_s_scaler = phi_loss_s_scaler
        self._psi_loss_scaler = psi_loss_scaler
        self._q_loss_scaler = q_loss_scaler

        # trainable networks
        self.convNet = convNet
        self.target_convNet = deepcopy(self.convNet)
        self.target_convNet.load_state_dict(self.convNet.state_dict())
        self.dcnNet = dcnNet
        self.qNet = qNet
        self.target_qNet = deepcopy(self.qNet)
        self.target_qNet.load_state_dict(self.qNet.state_dict())

        if options is not None:
            self._options = options
        else:
            self._options = nn.Parameter(
                torch.zeros(
                    1,
                    int(self._sf_dim / 2),
                    dtype=self._dtype,
                    device=self.device,
                    requires_grad=True,
                )
            ).to(self.device)

        self._target_options = self._options.clone().detach()

        self.optimizer = torch.optim.Adam(
            [
                {"params": convNet.parameters(), "lr": 5e-5},
                {"params": dcnNet.parameters(), "lr": 5e-5},
                {"params": qNet.parameters(), "lr": 5e-5},
                {"params": self._options, "lr": 5e-5},
            ]
        )

        #
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def change_mode(self, mode):
        self.decision_mode = mode

    def compute_q(self, phi):
        with torch.no_grad():
            psi = self.qNet(phi)
            psi_r, psi_s = torch.split(psi, psi.size(-1) // 2, dim=-1)

            q = self.multiply_options(
                psi_r, self._options
            ).squeeze()  # ~ [N, |A|, 1] -> [N, |A|]
        return q

    def forward(self, x, deterministic=False):
        self._forward_steps += 1
        # preprocessing
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x[None, :, :, :]
        x = x.to(self._dtype).to(self.device)

        # features
        phi = self.convNet(x)
        phi_r, phi_s = torch.split(phi, phi.size(-1) // 2, dim=-1)

        # actions
        if deterministic:
            q = self.compute_q(phi)
        else:
            if self.decision_mode == "random":
                q = torch.rand((1, self._a_dim)).to(self.device)
            elif self.decision_mode == "e-greedy":
                is_greedy = np.random.rand() >= self._epsilon
                q = (
                    self.compute_q(phi)
                    if is_greedy
                    else torch.rand((1, self._a_dim)).to(self.device)
                )
            else:
                raise NotImplementedError(
                    f"{self.decision_mode} is not implemented. Select among ['random', 'e-greedy']"
                )
        a = torch.argmax(q, dim=-1)
        a_oh = nn.functional.one_hot(a, num_classes=self._a_dim).to(
            self._dtype
        )  # onehot command gives output as Long type

        if self._epsilon >= 0 and self._anneal is not None:
            self._epsilon = self._epsilon - self._anneal * self._forward_steps

        return a, {
            "q": torch.mean(self.compute_q(phi), axis=-1),
            "phi": phi,
            "phi_r": phi_r,
            "phi_s": phi_s,
            "one_hot_action": a_oh,
        }

    def stateLoss(self, statePred, next_states):
        statePredLoss = self.mqe_loss(next_states, statePred)
        totalLoss = self._phi_loss_s_scaler * statePredLoss
        return totalLoss

    def rewardLoss(self, rewardPred, reward):
        rewardPredLoss = self.mse_loss(reward, rewardPred)
        totalLoss = self._phi_loss_r_scaler * rewardPredLoss
        return totalLoss

    def psiLoss(self, target_phi, psi, psiNext, actions, qPredNext):
        # [N, |A|, F] -> [N, F]
        filteredPsi = torch.sum(psi * actions.unsqueeze(-1), axis=1)

        # [N, |A|, F] -> [N, F]
        psiNextMax = torch.mean(psiNext, axis=1)

        td_target = (target_phi + self._gamma * psiNextMax).detach()
        psiLoss = torch.mean(self.mse_loss(td_target, filteredPsi))

        totalLoss = self._psi_loss_scaler * psiLoss

        return totalLoss

    def qLoss(self, reward, qPred, qPredNext, actions, terminals):
        # [N, |A|] -> [N, 1]
        curr_q = torch.sum(torch.mul(qPred, actions), axis=-1, keepdim=True)

        # [N, |A|] -> [N, 1]
        max_next_q = torch.max(qPredNext, axis=-1, keepdim=True)[0]

        td_target = reward + self._gamma * max_next_q * (
            1.0 - terminals.to(self._dtype)
        )

        qLoss = self.mse_loss(td_target, curr_q)
        totalLoss = self._q_loss_scaler * qLoss

        return totalLoss

    def run_network(self, states, actions, is_target=False):
        if is_target:
            phi = self.target_convNet(states)
            phi_r, phi_s = torch.split(phi, phi.size(-1) // 2, dim=-1)
            rewardPred = torch.sum(phi_r * self._target_options, axis=-1, keepdim=True)
            statePred = self.target_convNet.decode(phi_s, actions)

            psi = self.target_qNet(phi.detach())
            psi_r, psi_s = torch.split(psi, psi.size(-1) // 2, dim=-1)
            qPred = torch.sum(psi_r * self._target_options, axis=-1)
        else:
            phi = self.convNet(states)
            target_phi = self.target_convNet(states)
            phi_r, phi_s = torch.split(phi, phi.size(-1) // 2, dim=-1)
            rewardPred = torch.sum(phi_r * self._options, axis=-1, keepdim=True)
            statePred = self.convNet.decode(phi_s, actions)

            psi = self.qNet(phi.detach())
            psi_r, psi_s = torch.split(psi, psi.size(-1) // 2, dim=-1)
            qPred = torch.sum(psi_r * self._options, axis=-1)

        out = {
            "phi": phi,
            "target_phi": target_phi,
            "psi": psi,
            "w": self._options,
            "statePred": statePred,
            "rewardPred": rewardPred,
            "qPred": qPred,
        }

        return out

    def sf_learn(self, buffer):
        t0 = time.time()
        for _ in range(self._update_iter):
            buffer_batch = buffer.sample(self._trj_per_iter)
            states, actions, next_states, rewards, terminals = self.preprocess_batch(
                buffer_batch, self.device
            )
            out = self.run_network(states, actions)
            phi = out["phi"]
            target_phi = out["target_phi"]
            psi = out["psi"]
            w = out["w"]
            statePred = out["statePred"]
            rewardPred = out["rewardPred"]
            qPred = out["qPred"]

            out = self.run_network(next_states, actions, is_target=True)
            psiNext = out["psi"]
            qPredNext = out["qPred"]

            stateLoss = self.stateLoss(statePred, next_states)
            rewardLoss = self.rewardLoss(rewardPred, rewards)
            psiLoss = self.psiLoss(target_phi, psi, psiNext, actions, qPredNext)
            qLoss = self.qLoss(rewards, qPred, qPredNext, actions, terminals)

            loss = stateLoss + rewardLoss + psiLoss + qLoss

            self.optimizer.zero_grad()
            loss.backward()
            # self.qNet.check_grad()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=20.0)
            norm_dict = self.compute_weight_norm()
            grad_dict = self.compute_gradient_norm()
            self.optimizer.step()
            # self.qNet.check_param()

        # after update we sync the weight/bias
        self.target_convNet.load_state_dict(self.convNet.state_dict())
        self.target_qNet.load_state_dict(self.qNet.state_dict())
        self._target_options = self._options.clone().detach()

        loss_dict = {
            "loss": loss.item(),
            "phi_r_loss": rewardLoss.item(),
            "phi_s_loss": stateLoss.item(),
            "psi_loss": psiLoss.item(),
            "q_loss": qLoss.item(),
        }
        loss_dict.update(norm_dict)
        loss_dict.update(grad_dict)

        t1 = time.time()
        return loss_dict, t1 - t0

    def discover_options(self, batch, method="SVD"):
        with torch.no_grad():
            states = torch.from_numpy(batch["states"]).to(self.device)
            features = self.convNet(states)
            psi = self.qNet(features)
            psi = torch.mean(psi, axis=1)
            psi_r, psi_s = torch.split(psi, psi.size(-1) // 2, dim=-1)
        if method == "SVD":
            U, S, V = torch.svd(psi_s)  # S: max - min

            return S.to("cpu"), V.to("cpu")
        else:
            return 0, 0
