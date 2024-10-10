import time
import os
import cv2
import pickle
import numpy as np
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

from utils.utils import estimate_psi
from models.layers import MLP, ConvNetwork, PsiCritic
from models.policy.base_policy import BasePolicy

matplotlib.use("Agg")


def generate_2d_heatmap_image(Z, tile_size):
    # Create a 2D heatmap and save it as an image
    fig, ax = plt.subplots(figsize=(5, 5))

    # Example data for 2D heatmap
    vector_length = Z.shape[0]
    grid_size = int(np.sqrt(vector_length))

    if grid_size**2 != vector_length:
        raise ValueError(
            "The length of the eigenvector must be a perfect square to reshape into a grid."
        )

    Z = Z.reshape((grid_size, grid_size))

    norm_Z = np.linalg.norm(Z)
    # Plot heatmap
    heatmap = ax.imshow(Z, cmap="binary", aspect="auto")
    fig.colorbar(heatmap, ax=ax, shrink=0.5, aspect=5)

    ax.set_title(f"Norm of Z: {norm_Z:.2f}", pad=20)

    # Save the heatmap to a file
    id = str(uuid.uuid4())
    file_name = f"temp/{id}.png"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Read the saved image
    plot_img = cv2.imread(file_name)
    os.remove(file_name)
    plot_img = cv2.resize(
        plot_img, (7 * tile_size, 7 * tile_size)
    )  # Resize to match frame height
    return plot_img


class SF_Combined(BasePolicy):
    def __init__(
        self,
        feaNet: ConvNetwork,
        psiNet: PsiCritic,
        a_dim: int,
        options=None,
        feature_lr: float = 3e-4,
        psi_lr: float = 5e-4,
        option_lr: float = 1e-4,
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
        super(SF_Combined, self).__init__()

        # constants
        self.decision_mode = decision_mode
        self.device = device

        self._trj_per_iter = trj_per_iter
        self._a_dim = a_dim
        self._fc_dim = feaNet._fc_dim
        self._sf_dim = feaNet._sf_dim
        self._epsilon = epsilon
        self._gamma = gamma
        self._anneal = anneal
        self._forward_steps = 0
        self._phi_loss_r_scaler = phi_loss_r_scaler
        self._phi_loss_s_scaler = phi_loss_s_scaler
        self._psi_loss_scaler = psi_loss_scaler
        self._q_loss_scaler = q_loss_scaler

        # trainable networks
        self.feaNet = feaNet
        self.psiNet = psiNet

        if options is not None:
            self._options = options
        else:
            self._options = nn.Parameter(
                torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=(1, int(self._sf_dim)),
                    dtype=self._dtype,
                    device=self.device,
                )
            ).to(self.device)

        self.feature_optims = torch.optim.AdamW(
            [
                {"params": self.feaNet.parameters(), "lr": feature_lr},
                {"params": self._options, "lr": option_lr},
            ]
        )

        self.psi_optim = torch.optim.AdamW(params=self.psiNet.parameters(), lr=psi_lr)

        #
        self.dummy = torch.tensor(0.0)
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def change_mode(self, mode):
        self.decision_mode = mode

    def compute_q(self, phi):
        with torch.no_grad():
            psi, _ = self.psiNet(phi)

            q = self.multiply_options(
                psi, self._options
            ).squeeze()  # ~ [N, |A|, 1] -> [N, |A|]
        return q

    def forward(self, obs, z=None, deterministic=False):
        self._forward_steps += 1
        x = obs["observation"]
        agent_pos = obs["agent_pos"]

        # preprocessing
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(agent_pos, np.ndarray):
            agent_pos = torch.from_numpy(agent_pos)
        if len(x.shape) == 3:
            x = x[None, :, :, :]
        if len(agent_pos.shape) == 1:
            agent_pos = agent_pos[None, :]
        x = x.to(self._dtype).to(self.device)

        # features
        with torch.no_grad():
            phi, conv_dict = self.feaNet(x, agent_pos)

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
        a_oh = F.one_hot(a.long(), num_classes=self._a_dim).to(self._dtype)

        if self._epsilon >= 0 and self._anneal is not None:
            self._epsilon = self._epsilon - self._anneal * self._forward_steps

        return a, {
            "q": torch.mean(self.compute_q(phi), axis=-1),  # for plotting
            "a_oh": a_oh,
            "phi": phi,  # for plotting
            "phi_r": phi,  # for plotting
            "phi_s": phi,  # for plotting
            "conv_dict": conv_dict,
            "z": 0,  # dummy
            "termination": True,  # dummy
            "is_option": False,  # dummy
            "probs": self.dummy,  # dummy
            "logprobs": self.dummy,  # dummy
        }

    def _phi_Loss(self, states, actions, next_states, agent_pos, rewards):
        """
        Training target: phi_r (reward), phi_s (state)  -->  (Critic: feaNet)
        Method: reward mse (r - phi_r * w), state_pred mse (s' - D(phi_s, a))
        ---------------------------------------------------------------------------
        phi ~ [N, F/2]
        w ~ [1, F/2]
        """
        phi, conv_dict = self.feaNet(states, agent_pos, deterministic=False)

        state_pred = self.feaNet.decode(phi, actions, conv_dict)
        phi_s_loss = self._phi_loss_s_scaler * self.mqe_loss(next_states, state_pred)

        l2_norm = 0
        for param in self.feaNet.parameters():
            if param.requires_grad:  # Only include parameters that require gradients
                l2_norm += torch.norm(param, p=2)  # L

        phi_loss = conv_dict["loss"] + phi_s_loss + 1e-6 * l2_norm

        phi_norm = torch.norm(phi.detach())
        return phi_loss, {
            "phi": phi,
            "loss": conv_dict["loss"],
            "phi_r_loss": self.dummy,
            "phi_s_loss": phi_s_loss,
            "phi_norm": phi_norm,
            "phi_regul": 1e-6 * l2_norm,
        }

    def _psi_Loss(self, features, actions, terminals):
        """
        Training target: psi  -->  (Critic: psi_advantage, psi_state)
        Method: reducing TD error
        ---------------------------------------------------------------------------
        phi ~ [N, F]
        actions ~ [N, |A|]
        psi ~ [N, |A|, F]
        w ~ [1, F/2]
        """
        psi, _ = self.psiNet(features)

        filteredPsi = torch.sum(
            psi * actions.unsqueeze(-1), axis=1
        )  # -> filteredPsi ~ [N, F] since no keepdim=True

        psi_est = estimate_psi(features, terminals, self._gamma, self.device)
        psi_loss = self._psi_loss_scaler * self.huber_loss(psi_est, filteredPsi)

        l2_norm = 0
        for param in self.psiNet.parameters():
            if param.requires_grad:  # Only include parameters that require gradients
                l2_norm += torch.norm(param, p=2)  # L
        psi_loss += 1e-5 * l2_norm

        psi_norm = torch.norm(filteredPsi.detach())
        return psi_loss, {"psi_norm": psi_norm}

    # def _q_Loss(self, states, next_states, actions, rewards, terminals):
    #     """
    #     Training target: w (to stabilize phi_r loss)  -->  w (trainable tensors)
    #     Method: Q TD learning
    #     ---------------------------------------------------------------------------
    #     q ~ [N, |A|]
    #     w ~ [1, F/2]
    #     """
    #     with torch.no_grad():
    #         phi = self.convNet(states)

    #         next_phi = self.target_convNet(next_states)
    #         next_psi, _ = self.target_qNet(next_phi)

    #     psi, _ = self.qNet(phi)
    #     # ~ [N, |A|, 1] -> [N, |A|]
    #     q = self.multiply_options(psi, self._options).squeeze()

    #     # dimTrue~ [N,] -> [N, 1]
    #     curr_q = torch.sum(torch.mul(q, actions), axis=-1, keepdim=True)

    #     target_next_q = self.multiply_options(next_psi, self._target_options).squeeze()
    #     # ~ [N,] -> [N, 1]
    #     max_next_q = torch.max(target_next_q, axis=-1, keepdim=True)[0]

    #     ### could be trained with target q
    #     td_target = (
    #         rewards + self._gamma * max_next_q * (1 - terminals).to(self._dtype)
    #     ).detach()

    #     q_loss = self._q_loss_scaler * self.mqe_loss(td_target, curr_q)

    #     w_norm = torch.norm(self._options.detach())
    #     return q_loss, {"w_norm": w_norm}

    def learn(self, buffer):
        self.train()
        t0 = time.time()

        buffer_batch = buffer.sample(self._trj_per_iter)
        (
            states,
            _,
            _,
            actions_oh,
            next_states,
            agent_pos,
            next_agent_pos,
            rewards,
            _,
            _,
        ) = self.preprocess_batch(buffer_batch, self.device)

        phi_loss, phi_loss_dict = self._phi_Loss(
            states, actions_oh, next_states, agent_pos, rewards
        )

        self.feature_optims.zero_grad()
        phi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        phi_grad_dict = self.compute_gradient_norm(
            [self.feaNet, self._options],
            ["feaNet", "options"],
            dir="SF",
            device=self.device,
        )
        self.feature_optims.step()

        norm_dict = self.compute_weight_norm(
            [self.feaNet, self.psiNet, self._options],
            ["feaNet", "psiNet", "options"],
            dir="SF",
            device=self.device,
        )

        loss_dict = {
            "SF/loss": phi_loss.item(),
            "SF/kl_loss": phi_loss_dict["loss"].item(),
            "SF/phi_r_loss": phi_loss_dict["phi_r_loss"].item(),
            "SF/phi_s_loss": phi_loss_dict["phi_s_loss"].item(),
            "SF/phiOutNorm": phi_loss_dict["phi_norm"].item(),
            "SF/phiParamLoss": phi_loss_dict["phi_regul"].item(),
        }
        loss_dict.update(norm_dict)
        loss_dict.update(phi_grad_dict)

        t1 = time.time()
        self.eval()
        return loss_dict, t1 - t0

    def learnPsi(self, batch):
        self.train()
        t0 = time.time()

        _, features, actions_oh, _, _, terminals, _ = self.preprocess_batch(
            batch, self.device
        )

        psi_loss, psi_loss_dict = self._psi_Loss(features, actions_oh, terminals)

        post_loss = psi_loss  # + q_loss

        self.psi_optim.zero_grad()
        post_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        post_grad_dict = self.compute_gradient_norm(
            [self.psiNet],
            ["psiNet"],
            dir="SF",
            device=self.device,
        )
        self.psi_optim.step()

        norm_dict = self.compute_weight_norm(
            [self.feaNet, self.psiNet, self._options],
            ["feaNet", "psiNet", "options"],
            dir="SF",
            device=self.device,
        )

        loss_dict = {
            "SF/loss": post_loss.item(),
            "SF/psi_loss": psi_loss.item(),
            "SF/psiOutNorm": psi_loss_dict["psi_norm"].item(),
        }
        loss_dict.update(norm_dict)
        loss_dict.update(post_grad_dict)

        t1 = time.time()
        self.eval()
        return loss_dict, t1 - t0

    def save_model(self, logdir, epoch=None, is_best=False):
        self.feaNet = self.feaNet.cpu()
        self.psiNet = self.psiNet.cpu()
        options = nn.Parameter(self._options.clone().cpu())

        # save checkpoint
        if is_best:
            path = os.path.join(logdir, "best_model.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")
        pickle.dump(
            (self.feaNet, self.psiNet, options),
            open(path, "wb"),
        )

        self.feaNet = self.feaNet.to(self.device)
        self.psiNet = self.psiNet.to(self.device)
