import numpy as np
import torch
import torch.nn as nn

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
        action_dim,
        fc_dim: int = 256,
        sf_dim: int = 256,
        decoder_inpuit_dim: int = 256,
        activation: nn.Module = nn.ReLU(),
    ):
        super(ConvNetwork, self).__init__()

        s_dim, _, in_channels = state_dim

        # Parameters
        self._dtype = torch.float32
        self._fc_dim = fc_dim
        self._sf_dim = sf_dim

        # Activation functions
        self.act = activation

        ### Encoding module
        self.en_pmt = Permute((0, 3, 1, 2))

        ### conv structure
        conv_layers = [
            {
                "type": "conv",
                "kernel_size": 4,
                "stride": 2,
                "padding": 2,
                "activation": nn.Tanh(),
                "in_filters": in_channels,
                "out_filters": 16,
            },  # Halve the spatial dimensions
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": nn.Tanh(),
                "in_filters": 16,
                "out_filters": 32,
            },  # Halve spatial dimensions again
            # {
            #     "type": "pool",
            #     "kernel_size": 2,
            #     "stride": 1,
            #     "padding": 1,
            # },  # Max Pooling, reduce spatial dimensions by half
            {
                "type": "conv",
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "activation": nn.Tanh(),
                "in_filters": 32,
                "out_filters": 64,
            },  # Halve spatial dimensions again
            # {
            #     "type": "pool",
            #     "kernel_size": 2,
            #     "stride": 1,
            #     "padding": 1,
            # },  # Max Pooling, reduce spatial dimensions by half
            # {
            #     "type": "conv",
            #     "kernel_size": 3,
            #     "stride": 1,
            #     "padding": 0,
            #     "activation": nn.Tanh(),
            #     "in_filters": 32,
            #     "out_filters": 64,
            # },  # Halve spatial dimensions again
            # {
            #     "type": "pool",
            #     "kernel_size": 2,
            #     "stride": 1,
            #     "padding": 0,
            # },  # Max Pooling, reduce spatial dimensions by half
        ]

        results = check_output_padding_needed(conv_layers, s_dim)
        # Print the results

        self.output_paddings = [x["output_padding"] for x in results][::-1]

        # Define the fully connected layers
        flat_dim, output_shape = calculate_flatten_size(state_dim, conv_layers)
        reduced_feature_dim = output_shape[0]

        self.conv = nn.ModuleList()
        for layer in conv_layers:
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

        self.en_feature = MLP(
            input_dim=flat_dim,
            hidden_dims=(fc_dim, fc_dim, fc_dim),
            output_dim=sf_dim,
            activation=self.act,
        )

        # self.en_last_act = nn.ReLU()
        # self.en_last_act = nn.Tanh()
        self.en_last_act = nn.Sigmoid()
        # self.en_last_act = nn.Identity()
        # self.en_last_act = EncoderLastAct(alpha=1.0)

        ### Decoding module
        # preprocess
        self.de_action = MLP(
            input_dim=action_dim, hidden_dims=(fc_dim, fc_dim), activation=self.act
        )

        self.de_state_feature = MLP(
            input_dim=decoder_inpuit_dim,
            hidden_dims=(fc_dim, fc_dim),
            activation=self.act,
        )
        # self.de_state_feature = MLP(
        #     input_dim=sf_dim, hidden_dims=(fc_dim,), activation=self.act
        # )

        # main decoding module
        self.de_concat = MLP(
            input_dim=2 * fc_dim, hidden_dims=(flat_dim,), activation=self.act
        )

        self.reshape = Reshape(fc_dim, reduced_feature_dim)

        self.de_conv = nn.ModuleList()
        i = 0
        for layer in conv_layers[::-1]:
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
        # self.de_last_act = nn.Tanh()
        # self.de_last_act = nn.Sigmoid()
        # self.de_last_act = nn.Identity()
        # self.de_last_act = EncoderLastAct(alpha=1.0)

        self.de_pmt = Permute((0, 2, 3, 1))

    def forward(self, x, deterministic=True):
        indices = []
        sizes = []

        out = self.en_pmt(x)
        # print(out.shape)
        for fn in self.conv:
            output_dim = out.shape
            out, info = fn(out)
            if isinstance(fn, nn.MaxPool2d):
                indices.append(info)
                sizes.append(output_dim)
            # print(out.shape)
        out = self.en_flatter(out)
        # print(out.shape)
        out = self.en_feature(out)
        # print(out.shape)
        out = self.en_last_act(out)
        # print(out.shape)
        return out, {"indices": indices, "output_dim": sizes, "loss": torch.tensor(0.0)}

    def decode(self, features, actions, conv_dict):
        """This reconstruct full state given phi_state and actions"""
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(self._dtype)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self._dtype)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

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
            # print(out.shape)
        out = self.de_last_act(out)
        # print(out.shape)
        out = self.de_pmt(out)
        # print(out.shape)
        return out


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
        activation: nn.Module = nn.ReLU(),
    ):
        super(VAE, self).__init__()

        first_dim, second_dim, in_channels = state_dim
        input_dim = int(first_dim * second_dim * in_channels)

        # Parameters
        self._dtype = torch.float32
        self._fc_dim = fc_dim
        self._sf_dim = sf_dim

        # Activation functions
        self.act = activation

        ### embedding
        # self.tensorEmbed = MLP(
        #     input_dim=input_dim,
        #     hidden_dims=(input_dim,),
        #     activation=nn.Tanh(),
        # )
        # self.actionEmbed = MLP(
        #     input_dim=action_dim,
        #     hidden_dims=(action_dim,),
        #     activation=nn.Tanh(),
        # )

        ### Encoding module
        self.flatter = nn.Flatten()

        self.en_vae = MLP(
            input_dim=input_dim,
            hidden_dims=(fc_dim, fc_dim, int(fc_dim / 2)),
            activation=self.act,
        )

        # self.encoder = nn.Sequential(self.flatter, self.tensorEmbed, self.en_vae)
        self.encoder = nn.Sequential(self.flatter, self.en_vae)

        self.mu = MLP(
            input_dim=int(fc_dim / 2),
            hidden_dims=(fc_dim,),
            output_dim=sf_dim,
            activation=self.act,
        )
        self.logstd = MLP(
            input_dim=int(fc_dim / 2),
            hidden_dims=(fc_dim,),
            output_dim=sf_dim,
            activation=self.act,
        )

        ### Decoding module
        self.unflatter = Reshape(fc_dim, first_dim)

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
            hidden_dims=(fc_dim,),
            output_dim=input_dim,
            activation=self.act,
        )

        self.de_pmt = Permute((0, 2, 3, 1))

        self.decoder = nn.Sequential(
            self.concat, self.de_vae, self.unflatter, self.de_pmt
        )

    def forward(self, x, deterministic=True):
        """
        Input = x: 4D tensor arrays
        """
        if len(x.shape) == 1:
            x = x[None, :]
        out = self.flatter(x)
        out = self.encoder(out)

        mu = self.mu(out)
        logstd = self.logstd(out)

        std = torch.exp(logstd + 1e-7)
        cov = torch.diag_embed(std)

        dist = MultivariateNormal(loc=mu, covariance_matrix=cov)

        if deterministic:
            feature = mu
        else:
            feature = dist.rsample()

        kl = std**2 + mu**2 - torch.log(std) - 0.5
        kl_loss = torch.mean(torch.mean(kl, axis=-1), axis=-1)

        return feature, {"loss": kl_loss}

    def decode(self, features, actions, conv_dict):
        """This reconstruct full state given phi_state and actions"""
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(self._dtype)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self._dtype)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

        # actions = self.actionEmbed(actions)
        out1 = self.de_action(actions)
        out2 = self.de_latent(features)

        out = torch.cat((out1, out2), axis=-1)

        out = self.decoder(out)
        # print(out.shape)
        return out


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
        self._dtype = torch.float32

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
