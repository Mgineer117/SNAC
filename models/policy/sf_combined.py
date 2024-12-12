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
from models.layers import MLP, ConvNetwork, VAE, PsiCritic
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
        phi_loss_kl_scaler: float = 25.0,
        phi_loss_l2_scaler: float = 1e-6,
        psi_loss_scaler: float = 1.0,
        psi_loss_l2_scaler: float = 1e-6,
        q_loss_scaler: float = 0.0,
        is_discrete: bool = False,
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
        self._phi_loss_kl_scaler = phi_loss_kl_scaler
        self._phi_loss_l2_scaler = phi_loss_l2_scaler
        self._psi_loss_scaler = psi_loss_scaler
        self._psi_loss_l2_scaler = psi_loss_l2_scaler
        self._q_loss_scaler = q_loss_scaler

        self.is_discrete = is_discrete

        # trainable networks
        self.feaNet = feaNet
        self.psiNet = psiNet

        if options is not None:
            self._options = options
        else:
            self._options = nn.Parameter(
                torch.zeros(
                    size=(1, int(self._sf_dim)),
                    dtype=self._dtype,
                    device=self.device,
                )
            ).to(self.device)

        self.feature_optims = torch.optim.Adam(
            [
                {"params": self.feaNet.parameters(), "lr": feature_lr},
                {"params": self._options, "lr": option_lr},
            ]
        )
        self.psi_optim = torch.optim.Adam(params=self.psiNet.parameters(), lr=psi_lr)

        #
        self.dummy = torch.tensor(1e-5)
        self.to(self.device).to(self._dtype)

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

    def preprocess_obs(self, obs):
        observation = obs["observation"]
        agent_pos = obs["agent_pos"]

        if not torch.is_tensor(observation):
            observation = torch.from_numpy(observation).to(self._dtype).to(self.device)
        if agent_pos is not None and not torch.is_tensor(agent_pos):
            agent_pos = torch.from_numpy(agent_pos).to(self._dtype).to(self.device)

        return {"observation": observation, "agent_pos": agent_pos}

    def forward(self, obs, z=None, deterministic=False):
        self._forward_steps += 1
        obs = self.preprocess_obs(obs)

        if self.is_discrete:
            a = torch.randint(0, self._a_dim, (1,))
            a = F.one_hot(a, num_classes=self._a_dim)
        else:
            a = torch.rand((self._a_dim,))

        return a, {
            # some dummy variables to keep the code consistent across algs
            "z": self.dummy,  # dummy
            "probs": self.dummy,  # dummy
            "logprobs": self.dummy,  # dummy
            "entropy": self.dummy,  # dummy
        }

    def random_walk(self, obs):
        return self(obs)

    def get_features(self, obs, to_numpy: bool = False):
        obs = self.preprocess_obs(obs)
        with torch.no_grad():
            phi, _ = self.feaNet(
                obs["observation"], obs["agent_pos"], deterministic=True
            )
        if to_numpy:
            phi = phi.cpu().numpy()
        return phi, {}

    def get_cumulative_features(self, obs, to_numpy: bool = False):
        """
        The naming intuition is that phi and psi are not really distinguishable
        """
        obs = self.preprocess_obs(obs)
        with torch.no_grad():
            phi, _ = self.feaNet(
                obs["observation"], obs["agent_pos"], deterministic=True
            )
            psi, _ = self.psiNet(phi)

        if to_numpy:
            psi = psi.cpu().numpy()
        return psi, {}

    def decode(self, features, actions, conv_dict):
        # Does some dimensional and np <-> tensor work
        # and pass it to feature decoder actions should be one-hot
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(self.device).to(self._dtype)
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device).to(self._dtype)
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(0)

        reconstructed_state = self.feaNet.decode(features, actions, conv_dict)
        return reconstructed_state

    def _phi_Loss(self, states, actions, next_states, agent_pos, rewards):
        """
        Training target: phi_r (reward), phi_s (state)  -->  (Critic: feaNet)
        Method: reward mse (r - phi_r * w), state_pred mse (s' - D(phi_s, a))
        ---------------------------------------------------------------------------
        phi ~ [N, F/2]
        w ~ [1, F/2]
        """
        phi, conv_dict = self.feaNet(states, agent_pos, deterministic=False)

        state_pred = self.decode(phi, actions, conv_dict)
        if isinstance(self.feaNet, VAE):
            phi_s_loss = self._phi_loss_s_scaler * self.mse_loss(
                state_pred, next_states
            )
        else:
            phi_s_loss = self._phi_loss_s_scaler * self.mqe4D_loss(
                next_states, state_pred
            )

        option_loss_scaler = 1e-10
        option_loss = option_loss_scaler * ((1.0 - torch.norm(self._options, p=2)) ** 2)

        kl_loss = self._phi_loss_kl_scaler * conv_dict["loss"]

        l2_norm = 0
        for param in self.feaNet.parameters():
            if param.requires_grad:  # Only include parameters that require gradients
                l2_norm += torch.norm(param, p=2)  # L
        l2_loss = self._phi_loss_l2_scaler * l2_norm

        phi_loss = kl_loss + phi_s_loss + option_loss + l2_loss

        phi_norm = torch.norm(phi.detach())
        return phi_loss, {
            "phi": phi,
            "loss": kl_loss,
            "phi_r_loss": self.dummy,
            "phi_s_loss": phi_s_loss,
            "option_loss": option_loss,
            "phi_norm": phi_norm,
            "phi_regul": l2_loss,
        }

    def _psi_Loss(self, states, agent_pos, actions, terminals):
        """
        Training target: psi  -->  (Critic: psi_advantage, psi_state)
        Method: reducing TD error
        ---------------------------------------------------------------------------
        phi ~ [N, F]
        actions ~ [N, |A|]
        psi ~ [N, |A|, F]
        w ~ [1, F/2]
        """
        with torch.no_grad():
            phi, _ = self.feaNet(states, agent_pos)

        psi, _ = self.psiNet(phi)

        filteredPsi = torch.sum(
            psi * actions.unsqueeze(-1), axis=1
        )  # -> filteredPsi ~ [N, F] since no keepdim=True

        psi_est = estimate_psi(phi, terminals, self._gamma, self.device)
        psi_loss = self._psi_loss_scaler * self.mse_loss(psi_est, filteredPsi)

        l2_norm = 0
        for param in self.psiNet.parameters():
            if param.requires_grad:  # Only include parameters that require gradients
                l2_norm += torch.norm(param, p=2)  # L
        l2_loss = self._psi_loss_l2_scaler * l2_norm

        loss = psi_loss + l2_loss

        psi_norm = torch.norm(filteredPsi.detach())
        return loss, {"psi_loss": psi_loss, "psi_regul": l2_loss, "psi_norm": psi_norm}

    def learn(self, buffer):
        self.train()
        t0 = time.time()

        ### Pull data from the batch
        batch = buffer.sample(self._trj_per_iter)
        states = torch.from_numpy(batch["states"]).to(self._dtype).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self._dtype).to(self.device)
        next_states = (
            torch.from_numpy(batch["next_states"]).to(self._dtype).to(self.device)
        )
        agent_pos = torch.from_numpy(batch["agent_pos"]).to(self._dtype).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self._dtype).to(self.device)

        ### Compute the Loss
        phi_loss, phi_loss_dict = self._phi_Loss(
            states, actions, next_states, agent_pos, rewards
        )

        ### Update the network
        self.feature_optims.zero_grad()
        phi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        phi_grad_dict = self.compute_gradient_norm(
            [self.feaNet, self._options],
            ["feaNet", "options"],
            dir="SF",
            device=self.device,
        )
        norm_dict = self.compute_weight_norm(
            [self.feaNet, self.psiNet, self._options],
            ["feaNet", "psiNet", "options"],
            dir="SF",
            device=self.device,
        )
        self.feature_optims.step()

        loss_dict = {
            "SF/loss": phi_loss.item(),
            "SF/kl_loss": phi_loss_dict["loss"].item(),
            "SF/phi_r_loss": phi_loss_dict["phi_r_loss"].item(),
            "SF/phi_s_loss": phi_loss_dict["phi_s_loss"].item(),
            "SF/option_loss": phi_loss_dict["option_loss"].item(),
            "SF/phiOutNorm": phi_loss_dict["phi_norm"].item(),
            "SF/phiParamLoss": phi_loss_dict["phi_regul"].item(),
        }
        loss_dict.update(norm_dict)
        loss_dict.update(phi_grad_dict)

        t1 = time.time()
        self.eval()
        return loss_dict, t1 - t0

    def learnPsi(self, batch):
        """
        deprecared
        """

        self.train()
        t0 = time.time()

        ### Pull data from the batch
        states = torch.from_numpy(batch["states"]).to(self._dtype).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self._dtype).to(self.device)
        agent_pos = torch.from_numpy(batch["agent_pos"]).to(self._dtype).to(self.device)
        terminals = torch.from_numpy(batch["terminals"]).to(self._dtype).to(self.device)

        psi_loss, psi_loss_dict = self._psi_Loss(states, agent_pos, actions, terminals)

        self.psi_optim.zero_grad()
        psi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        grad_dict = self.compute_gradient_norm(
            [self.psiNet],
            ["psiNet"],
            dir="SF",
            device=self.device,
        )
        norm_dict = self.compute_weight_norm(
            [self.feaNet, self.psiNet, self._options],
            ["feaNet", "psiNet", "options"],
            dir="SF",
            device=self.device,
        )
        self.psi_optim.step()

        loss_dict = {
            "SF/loss": psi_loss.item(),
            "SF/psi_loss": psi_loss_dict["psi_loss"].item(),
            "SF/psi_l2_loss": psi_loss_dict["psi_regul"].item(),
            "SF/psiOutNorm": psi_loss_dict["psi_norm"].item(),
        }
        loss_dict.update(norm_dict)
        loss_dict.update(grad_dict)

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
