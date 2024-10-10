import time
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from utils.utils import estimate_advantages
from models.layers.building_blocks import MLP
from models.layers.sf_networks import ConvNetwork, PsiCritic
from models.layers.op_networks import OptionPolicy, OptionCritic
from models.policy.base_policy import BasePolicy


def check_all_devices(module):
    devices = {param.device for param in module.parameters()}  # Get all unique devices
    return devices


class RandomWalk(BasePolicy):
    def __init__(
        self,
        convNet: ConvNetwork,
        qNet: PsiCritic,
        options: nn.Module,
        a_dim: int,
        device: str = "cpu",
    ):
        super(RandomWalk, self).__init__()

        # constants
        self.device = device

        self._options = nn.Parameter(options.to(self._dtype).to(self.device))
        self._num_options = options.shape[0]
        self._a_dim = a_dim
        self._forward_steps = 0

        # trainable networks
        self.convNet = convNet
        self.qNet = qNet

        #
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def getPsi(self, x):
        """
        x is a raw state (image) [N, 7, 7, 3]
        """
        with torch.no_grad():
            phi, _ = self.convNet(x)
            psi, _ = self.qNet(phi)
        return psi, {"phi": phi}

    def forward(self, x, z, deterministic=False):
        self._forward_steps += 1
        """
        x is state ~ (7, 7, 3)
        Always determinstic
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if len(x.shape) == 3:
            x = x[None, :, :, :]
        x = x.to(self._dtype).to(self.device)

        psi, metaData = self.getPsi(x)

        q = torch.rand((x.shape[0], self._a_dim)).to(self.device)

        # if z == "random":
        #     q = torch.rand((x.shape[0], self._a_dim)).to(self.device)
        # else:
        #     psi_r, psi_s = self.split(psi)

        #     if z < (self._num_options / 2):
        #         q = self.multiply_options(psi_r, self._options[z, :]).squeeze()
        #     else:
        #         q = self.multiply_options(psi_s, self._options[z, :]).squeeze()

        termination = True if torch.all(q <= 0) else False

        a = torch.argmax(q, dim=-1)
        a_oh = F.one_hot(a.long(), num_classes=self._a_dim)

        probs = F.softmax(q, dim=-1)
        logprobs = F.log_softmax(q, dim=-1)

        probs = torch.sum(probs * a_oh, axis=-1)
        logprobs = torch.sum(logprobs * a_oh, axis=-1)

        return a, {
            "q": q,
            "phi": metaData["phi"],
            "is_option": False,  # dummy
            "z": 0.0,
            "termination": termination,
            "probs": probs,
            "logprobs": logprobs,
        }
