import torch
import torch.nn as nn
from typing import Dict, List, Union, Tuple, Optional


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation: nn.Module = nn.ReLU(),
        initialization: bool = True,
        dropout_rate: Optional[float] = None,
    ) -> None:
        super().__init__()

        model = []

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

        if initialization:
            nn.init.xavier_uniform_(conv_layer.weight)
            conv_layer.bias.data.fill_(0.01)

        model += [conv_layer, activation]

        if dropout_rate is not None:
            model += [nn.Dropout(p=dropout_rate)]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x), {}


class DeConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        activation: nn.Module = nn.ReLU(),
        initialization: bool = True,
        dropout_rate: Optional[float] = None,
    ) -> None:
        super().__init__()

        model = []

        de_conv_layer = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        if initialization:
            nn.init.xavier_uniform_(de_conv_layer.weight)
            de_conv_layer.bias.data.fill_(0.01)

        # since this is decon, activation first
        model += [activation, de_conv_layer]

        if dropout_rate is not None:
            model += [nn.Dropout(p=dropout_rate)]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x), {}


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        initialization: bool = True,
        activation: nn.Module = nn.ReLU(),
        dropout_rate: Optional[float] = None,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            linear_layer = nn.Linear(in_dim, out_dim)
            if initialization:
                nn.init.xavier_uniform_(linear_layer.weight)
                linear_layer.bias.data.fill_(0.01)
            model += (
                [linear_layer, activation] if activation is not None else [linear_layer]
            )

            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            linear_layer = nn.Linear(hidden_dims[-1], output_dim)
            nn.init.xavier_uniform_(linear_layer.weight)
            linear_layer.bias.data.fill_(0.01)

            model += [linear_layer]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
