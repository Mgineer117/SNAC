import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import MaxPool2d, MaxUnpool2d
from torch.distributions import MultivariateNormal
from utils.utils import calculate_flatten_size, check_output_padding_needed
from models.policy.module.dist_module import DiagGaussian
from models.policy.module.actor_module import ActorProb
from models.policy.module.critic_module import Critic
from models.layers.building_blocks import MLP, Conv, DeConv
from typing import Optional, Dict, List, Tuple


class Permute(nn.Module):
    """
    Given dimensions (0, 3, 1, 2), it permutes the tensors to given dim.
    It was created with nn.Module to create a sequential module using nn.Sequaltial()
    """

    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class Reshape(nn.Module):
    """
    Given dimension k in [N, k], it divides into [N, ?, 4, 4] where ? * 4 * 4 = k
    It was created with nn.Module to create a sequential module using nn.Sequaltial()
    """

    def __init__(self, fc_dim, reduced_feature_dim):
        super(Reshape, self).__init__()
        self.fc_dim = fc_dim
        self.reduced_feature_dim = reduced_feature_dim

    def forward(self, x):
        N = x.shape[0]
        if torch.numel(x) < N * self.reduced_feature_dim * self.reduced_feature_dim:
            return x.view(N, -1, self.reduced_feature_dim, 1)
        else:
            return x.view(N, -1, self.reduced_feature_dim, self.reduced_feature_dim)


class EncoderLastAct(nn.Module):
    """
    Given dimension k in [N, k], it divides into [N, ?, 4, 4] where ? * 4 * 4 = k
    It was created with nn.Module to create a sequential module using nn.Sequaltial()
    """

    def __init__(self, alpha):
        super(EncoderLastAct, self).__init__()
        self._alpha = alpha

    def forward(self, x):
        return torch.minimum(
            torch.tensor(self._alpha), torch.maximum(torch.tensor(0.0), x)
        )


class ConvNetwork(nn.Module):
    """
    State encoding module
    -----------------------------------------
    1. Define each specific layer for encoder and decoder
    2. Use nn.Sequential in the end to sequentialize each networks
    """

    def __init__(
        self,
        state_dim: tuple,
        action_dim: int,
        agent_num: int,
        grid_size: int,
        encoder_conv_layers: list,
        decoder_conv_layers: list,
        fc_dim: int = 256,
        sf_dim: int = 256,
        decoder_inpuit_dim: int = 256,
        activation: nn.Module = nn.ReLU(),
    ):
        super(ConvNetwork, self).__init__()

        s_dim, _, in_channels = state_dim

        # Parameters
        self._fc_dim = fc_dim
        self._sf_dim = sf_dim
        self._agent_num = agent_num
        self._grid_size = grid_size

        # Activation functions
        self.act = activation

        ### Encoding module
        self.en_pmt = Permute((0, 3, 1, 2))

        ### conv structure
        results = check_output_padding_needed(encoder_conv_layers, s_dim)
        # Print the results

        self.output_paddings = [x["output_padding"] for x in results][::-1]

        # Define the fully connected layers
        flat_dim, output_shape = calculate_flatten_size(state_dim, encoder_conv_layers)
        reduced_feature_dim = output_shape[0]

        self.conv = nn.ModuleList()
        for layer in encoder_conv_layers:
            if layer["type"] == "conv":
                element = Conv(
                    in_channels=in_channels,
                    out_channels=layer["out_filters"],
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                    activation=layer["activation"],
                )
                in_channels = layer["out_filters"]

            elif layer["type"] == "pool":
                element = MaxPool2d(
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                    return_indices=True,
                )
            self.conv.append(element)

        #
        self.en_flatter = torch.nn.Flatten()

        feature_input_dim = flat_dim + (self._grid_size * self._grid_size)
        self.en_feature = MLP(
            input_dim=feature_input_dim,  # agent pos concat
            hidden_dims=(feature_input_dim,),
            activation=self.act,
        )

        self.en_reward = MLP(
            input_dim=feature_input_dim,  # agent pos concat
            hidden_dims=(fc_dim, fc_dim),
            output_dim=int(sf_dim / 2),
            activation=self.act,
        )

        self.en_state = MLP(
            input_dim=feature_input_dim,  # agent pos concat
            hidden_dims=(fc_dim, fc_dim),
            output_dim=int(sf_dim / 2),
            activation=self.act,
        )

        self.en_last_act = nn.ReLU()
        # self.en_last_act = nn.ELU()
        # self.en_last_act = nn.Tanh()
        # self.en_last_act = nn.Sigmoid()
        # self.en_last_act = nn.Identity()
        # self.en_last_act = EncoderLastAct(alpha=1.0)

        ### Decoding module
        # preprocess
        self.de_action = MLP(
            input_dim=action_dim, hidden_dims=(fc_dim,), activation=self.act
        )

        self.de_state_feature = MLP(
            input_dim=decoder_inpuit_dim,
            hidden_dims=(fc_dim,),
            activation=self.act,
        )

        # main decoding module
        self.de_concat = MLP(
            input_dim=2 * fc_dim,
            hidden_dims=(
                2 * fc_dim,
                flat_dim,
            ),
            activation=self.act,
        )

        self.reshape = Reshape(fc_dim, reduced_feature_dim)

        self.de_conv = nn.ModuleList()
        i = 0
        for layer in decoder_conv_layers[::-1]:
            if layer["type"] == "conv":
                element = DeConv(
                    in_channels=in_channels,
                    out_channels=layer["in_filters"],
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                    output_padding=self.output_paddings[i],
                    activation=layer["activation"],
                )
                in_channels = layer["in_filters"]
                i += 1

            elif layer["type"] == "pool":
                element = MaxUnpool2d(
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                    padding=layer["padding"],
                )
            self.de_conv.append(element)

        self.de_last_act = nn.ReLU()
        # self.de_last_act = nn.ELU()
        # self.de_last_act = nn.Tanh()
        # self.de_last_act = nn.Sigmoid()
        # self.de_last_act = nn.Identity()
        # self.de_last_act = EncoderLastAct(alpha=1.0)

        self.de_pmt = Permute((0, 2, 3, 1))

    def pre_grad_cam(self, x: torch.Tensor):
        """
        For grad-cam to visualize the feature activation
        """
        out = self.en_pmt(x)
        for fn in self.conv:
            out, _ = fn(out)
        return out

    def post_grad_cam(self, state: torch.Tensor, agent_pos: torch.Tensor):
        """
        For grad-cam to visualize the feature activation
        """
        # Create one-hot encodings for the tensor along the last dimension
        # one_hot = torch.nn.functional.one_hot(
        #     agent_pos.long(), num_classes=self._grid_size
        # )
        # agent_pos = one_hot.view(agent_pos.size(0), -1)

        # Initialize an empty map on the same device
        agent_map = torch.zeros(
            (state.shape[0], self._grid_size, self._grid_size), device=agent_pos.device
        )

        # Creates spatial tensor map using agent pos
        for i in range(self._agent_num):
            x_pos = agent_pos[:, 2 * i].long()
            y_pos = agent_pos[:, 2 * i + 1].long()
            if x_pos == 0.0 and y_pos == 0.0:
                # for masking purpose in some cases
                value = 0.0
            else:
                value = 1 if i == 0 else -1  # Value for the agent or enemy
            agent_map[torch.arange(state.shape[0]), x_pos, y_pos] = value

        # Reshape the map
        agent_map = agent_map.reshape(state.shape[0], -1)

        out = self.en_flatter(state)
        out = torch.cat((out, agent_map), axis=-1)
        out = self.en_feature(out)
        r_out = self.en_reward(out)
        s_out = self.en_state(out)
        out = torch.cat((r_out, s_out), axis=-1)
        out = self.en_last_act(out)
        return out

    def forward(
        self,
        state: torch.Tensor,
        agent_pos: torch.Tensor,
        deterministic: bool | None = None,
    ):
        """forward eats in observation and agent_pos (informational supplement) to output the feature vector

        Args:
            state (torch.Tensor): 1D or 2d state of the environment
            agent_pos (torch.Tensor): the position of agent (supplement)
            deterministic (bool, optional): Not used but here exists for code consistency

        Returns:
            feature: latent representations of the given state
        """
        # Create one-hot encodings for the tensor along the last dimension
        # one_hot = torch.nn.functional.one_hot(
        #     agent_pos.long(), num_classes=self._grid_size
        # )
        # agent_pos = one_hot.view(agent_pos.size(0), -1)

        # dimensional work for images
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        if len(agent_pos.shape) == 1:
            agent_pos = agent_pos.unsqueeze(0)

        # Initialize an empty map on the same device
        agent_map = torch.zeros(
            (state.shape[0], self._grid_size, self._grid_size), device=agent_pos.device
        )

        # Creates spatial tensor map using agent pos
        for i in range(self._agent_num):
            x_pos = agent_pos[:, 2 * i].long()
            y_pos = agent_pos[:, 2 * i + 1].long()
            value = 1 if i == 0 else 10  # Value for the agent or enemy
            agent_map[torch.arange(state.shape[0]), x_pos, y_pos] = value

        # Reshape the map
        agent_map = agent_map.reshape(state.shape[0], -1)

        # forward method
        indices = []
        sizes = []

        out = self.en_pmt(state)

        for fn in self.conv:
            output_dim = out.shape
            out, info = fn(out)
            if isinstance(fn, nn.MaxPool2d):
                indices.append(info)
                sizes.append(output_dim)

        out = self.en_flatter(out)
        out = torch.cat((out, agent_map), axis=-1)
        out = self.en_feature(out)
        r_out = self.en_reward(out)
        s_out = self.en_state(out)
        out = torch.cat((r_out, s_out), axis=-1)
        features = self.en_last_act(out)
        return features, {
            "indices": indices,
            "output_dim": sizes,
            "loss": torch.tensor(0.0),
        }

    def decode(self, features, actions, conv_dict):
        """This reconstruct full state given phi_state and actions"""

        indices = conv_dict["indices"][::-1]  # indices should be backward
        output_dim = conv_dict["output_dim"][::-1]  # to keep dim correct

        features = self.de_state_feature(features)
        actions = self.de_action(actions)

        out = torch.cat((features, actions), axis=-1)
        out = self.de_concat(out)
        out = self.reshape(out)

        i = 0
        for fn in self.de_conv:
            if isinstance(fn, nn.MaxUnpool2d):
                out = fn(out, indices[i], output_size=output_dim[i])
                i += 1
            else:
                out, _ = fn(out)
        out = self.de_last_act(out)
        reconstructed_state = self.de_pmt(out)
        return reconstructed_state


class VAE(nn.Module):
    """
    State encoding module
    -----------------------------------------
    1. Define each specific layer for encoder and decoder
    2. Use nn.Sequential in the end to sequentialize each networks
    """

    def __init__(
        self,
        state_dim: tuple,
        action_dim: int,
        fc_dim: int = 256,
        sf_dim: int = 256,
        decoder_inpuit_dim: int = 256,
        is_snac: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        super(VAE, self).__init__()

        first_dim: int
        second_dim: int
        in_channels: int
        if len(state_dim) == 3:
            first_dim, second_dim, in_channels = state_dim
        elif len(state_dim) == 1:
            first_dim = state_dim[0]
            second_dim = 1
            in_channels = 1
        else:
            raise ValueError("State dimension is not correct.")

        input_dim = int(first_dim * second_dim * in_channels)

        # Parameters
        self._fc_dim = fc_dim
        self._sf_dim = sf_dim
        self._is_snac = is_snac

        self.logstd_range = (-10, 2)

        # Activation functions
        self.act = activation

        ### Encoding module
        self.flatter = nn.Flatten()

        self.en_vae = MLP(
            input_dim=input_dim,
            hidden_dims=(fc_dim, fc_dim, fc_dim),
            activation=self.act,
        )

        # self.encoder = nn.Sequential(self.flatter, self.tensorEmbed, self.en_vae)
        self.encoder = nn.Sequential(self.flatter, self.en_vae)

        self.mu = MLP(
            input_dim=fc_dim,
            hidden_dims=(fc_dim,),
            output_dim=sf_dim,
            activation=nn.Identity(),
        )
        self.logstd = MLP(
            input_dim=fc_dim,
            hidden_dims=(fc_dim,),
            output_dim=sf_dim,
            activation=nn.Identity(),
        )

        ### Decoding module
        self.de_latent = MLP(
            input_dim=decoder_inpuit_dim,
            hidden_dims=(fc_dim,),
            activation=self.act,
        )

        self.de_action = MLP(
            input_dim=action_dim,
            hidden_dims=(fc_dim,),
            activation=self.act,
        )

        self.concat = MLP(
            input_dim=int(2 * fc_dim),
            hidden_dims=(fc_dim,),
            activation=self.act,
        )

        self.de_vae = MLP(
            input_dim=fc_dim,
            hidden_dims=(fc_dim, fc_dim, fc_dim),
            output_dim=input_dim,
            activation=self.act,
        )

        self.decoder = nn.Sequential(self.concat, self.de_vae)

    def forward(self, state: torch.Tensor, agent_pos: None = None, deterministic=True):
        """
        Input = x: 1D or 2D tensor arrays
        """
        # 1D -> 2D for consistency
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        out = self.flatter(state)
        out = self.encoder(out)

        mu = F.tanh(self.mu(out))
        logstd = torch.clamp(
            self.logstd(out),
            min=self.logstd_range[0],
            max=self.logstd_range[1],
        )

        std = torch.exp(logstd)
        cov = torch.diag_embed(std**2)

        dist = MultivariateNormal(loc=mu, covariance_matrix=cov)

        feature = mu if deterministic else dist.rsample()

        if self._is_snac:
            # Sum over latent dimensions, then mean over batch
            # Only for spatial features
            dim_half = mu.size(-1) // 2  # Half the dimension
            mu_half = mu[..., :dim_half]
            std_half = std[..., :dim_half]

            kl = -0.5 * torch.sum(
                1 + torch.log(std_half**2) - mu_half**2 - std_half**2, dim=-1
            )
        else:
            # Sum over latent dimensions, then mean over batch
            kl = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=-1)
        kl_loss = torch.mean(kl)

        return feature, {"loss": kl_loss}

    def decode(self, features, actions, conv_dict: None = None):
        """This reconstruct full state given phi_state and actions"""
        out2 = self.de_latent(features)
        out1 = self.de_action(actions)

        out = torch.cat((out1, out2), axis=-1)
        reconstructed_state = self.decoder(out)
        return reconstructed_state


class PsiAdvantage(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self, fc_dim: int, sf_dim: int, a_dim: int, activation: nn.Module = nn.ReLU()
    ):
        super(PsiAdvantage, self).__init__()

        # |A| duplicate networks
        self.act = activation
        # ex_layer = self.create_sequential_model(fc_dim, sf_dim)

        self.models = nn.ModuleList()
        for _ in range(a_dim):
            self.models.append(self.create_sequential_model(fc_dim, sf_dim))

    def create_sequential_model(self, fc_dim, sf_dim):
        return MLP(sf_dim, (fc_dim, fc_dim), sf_dim, activation=self.act)

    def forward(self, x: torch.Tensor):
        X = []
        for model in self.models:
            X.append(model(x.clone()))
        X = torch.stack(X, dim=1)
        return X


class PsiCritic(nn.Module):
    """
    s
    """

    def __init__(
        self, fc_dim: int, sf_dim: int, a_dim: int, activation: nn.Module = nn.ReLU()
    ):
        super(PsiCritic, self).__init__()

        # Algorithmic parameters
        self.act = activation
        self._a_dim = a_dim

        self.psi_advantage = PsiAdvantage(fc_dim, sf_dim, a_dim, self.act)
        self.psi_state = MLP(
            input_dim=sf_dim,
            hidden_dims=(fc_dim, fc_dim),
            output_dim=sf_dim,
            activation=self.act,
        )

    def forward(self, x, z=None):
        """
        x: phi
        phi = (phi_r, phi_s)
        psi = (psi_r, psi_s)
        Q = psi_r * w where w = eig(psi_s)
        ------------------------------------------
        Previous method of Q = psi_s * w where w = eig(psi_s) aims to navigate to 'bottleneck' while that may not be a goal
        therefore we need to modify the Q-direction by projecting onto the reward space.
        """
        psi_advantage = self.psi_advantage(x)
        psi_state = self.psi_state(x)

        psi = (
            psi_state.unsqueeze(1)
            + psi_advantage
            - torch.mean(psi_advantage, axis=1, keepdim=True)
        )  # psi ~ [N, |A|, F]

        # psi_r, psi_s = torch.split(psi, psi.size(-1) // 2, dim=-1)

        return psi, {"psiState": psi_state, "psiAdvantage": psi_advantage}

    def check_param(self):
        shared_grad = False
        num_models = len(self.psi_advantage.models)

        # Check gradients after backward pass
        for i in range(num_models):
            for j in range(i + 1, num_models):
                model1 = self.psi_advantage.models[i]
                model2 = self.psi_advantage.models[j]

                # Check each parameter's gradient in model1 against model2
                for (name1, param1), (name2, param2) in zip(
                    model1.named_parameters(), model2.named_parameters()
                ):
                    if param1.grad is not None and param2.grad is not None:
                        if torch.equal(param1, param2):
                            print(
                                f"Models {i} and {j} have the same param for parameter: {name1}"
                            )
                            shared_grad = True

        if not shared_grad:
            print("No models have the same params.")

        return shared_grad

    def check_sharing(self):
        shared = False
        num_models = len(self.psi_advantage.models)

        # Compare each model with every other model
        for i in range(num_models):
            for j in range(i + 1, num_models):
                model1 = self.psi_advantage.models[i]
                model2 = self.psi_advantage.models[j]

                # Check each parameter in model1 against model2
                for (name1, param1), (name2, param2) in zip(
                    model1.named_parameters(), model2.named_parameters()
                ):
                    if id(param1) == id(param2):
                        print(f"Models {i} and {j} share parameter: {name1}")
                        shared = True

        if not shared:
            print("No shared parameters found among the models.")

        return shared

    def check_grad(self):
        shared_grad = False
        num_models = len(self.psi_advantage.models)

        # Check gradients after backward pass
        for i in range(num_models):
            for j in range(i + 1, num_models):
                model1 = self.psi_advantage.models[i]
                model2 = self.psi_advantage.models[j]

                # Check each parameter's gradient in model1 against model2
                for (name1, param1), (name2, param2) in zip(
                    model1.named_parameters(), model2.named_parameters()
                ):
                    if param1.grad is not None and param2.grad is not None:
                        if torch.equal(param1.grad, param2.grad):
                            print(
                                f"Models {i} and {j} have the same gradient for parameter: {name1}"
                            )
                            shared_grad = True

        if not shared_grad:
            print("No models have the same gradients.")

        return shared_grad
