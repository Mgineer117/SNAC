import torch
import torch.nn as nn
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from models.evaulators.base_evaluator import DotDict
from algorithms.SNAC import SNAC
from algorithms.EigenOption import EigenOption
from algorithms.CoveringOption import CoveringOption
from algorithms.PPO import PPO
from algorithms.FeatureTrain import FeatureTrain
from gym_multigrid.envs.fourrooms import FourRooms
from gym_multigrid.envs.ctf import Ctf1v1Env

from utils import *

import wandb

wandb.require("core")


class GradCam(nn.Module):
    def __init__(self, sf_network, algo_name):
        super(GradCam, self).__init__()

        # get the pretrained VGG19 network
        self.feaNet = sf_network.feaNet
        self.preGrad = sf_network.feaNet.pre_grad_cam
        self.postGrad = sf_network.feaNet.post_grad_cam

        self.algo_name = algo_name

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, target="s"):
        x = self.preGrad(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        x = self.postGrad(x)

        # apply the remaining pooling
        if self.algo_name == "SNAC":
            x_r, x_s = torch.split(x, x.size(-1) // 2, dim=-1)
            x = torch.sum(x_s, dim=-1)
            if target == "s":
                x = torch.sum(x_s, dim=-1)
            elif target == "r":
                x = torch.sum(x_r, dim=-1)
            else:
                raise ValueError(f"Unknown target: {target}")
        else:
            x = torch.sum(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.preGrad(x)


def get_env(args):
    args.grid_size = 12
    args.a_dim = 5
    args.draw_map = False
    map_path: str = "assets/ctf_avoid_obj.txt"
    observation_option: str = "tensor"
    if args.env_name == "CtF1v1":
        env = Ctf1v1Env(map_path=map_path, observation_option=observation_option)
    else:
        raise NotImplementedError(f"{args.env_name} not implemented")
    env = NoStateDictCtfWrapper(env, tile_size=args.tile_size)
    return env


def get_grid(env, args):
    grid, (x_coords, y_coords), loc = get_grid_tensor2(env, args.env_seed)
    grid = grid[None, :, :, :]
    pos = np.hstack((x_coords[:, None], y_coords[:, None]))

    grid = torch.from_numpy(grid).to(torch.float32)
    pos = torch.from_numpy(pos).to(torch.float32)
    return grid, pos


def plot(img, heatmap, i):

    ### Figure
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # img = img**5
    # axes[0].imshow(img[0, :, :, :])
    # axes[0].set_title("Original")

    # # Plot the second heatmap
    # axes[1].matshow(heatmap)
    # axes[1].set_title("Heatmap")
    # axes[1].colorbar = plt.colorbar(plt.cm.ScalarMappable(), ax=axes[1])

    # # draw the heatmap
    # plt.axis("off")
    # plt.savefig(f"heatmap/{i}.png")
    # print(f"{i} th figure saved")
    # plt.close()

    img = img.numpy()
    for j in range(img.shape[-1]):
        min_val = img[0, :, :, j].min()
        max_val = img[0, :, :, j].max()
        img[0, :, :, j] = (img[0, :, :, j] - min_val) / (max_val - min_val + 1e-7)

    heatmap = heatmap.numpy()

    ### Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img[0, :, :, :])
    axes[0].set_title("Original")

    # Plot the second heatmap
    axes[1].matshow(heatmap)
    axes[1].set_title("Heatmap")
    axes[1].colorbar = plt.colorbar(plt.cm.ScalarMappable(), ax=axes[1])

    # Plot the second heatmap
    alpha = 0.5
    heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap / 255)
    superimposed_img = heatmap * alpha + img[0, :, :, :] * (1 - alpha)

    axes[2].matshow(superimposed_img)
    axes[2].set_title("Super-imposed")

    # draw the heatmap
    plt.axis("off")
    plt.savefig(f"heatmap/{i}.png")
    print(f"{i} th figure saved")
    plt.close()


def run_loop(grid, pos, target="s"):
    for i in range(pos.shape[0]):
        x, y = pos[i, 0], pos[i, 1]

        img = grid.clone()

        img[:, x.long(), y.long(), 1] = 1

        # do grad-cam
        out = gradCam(img, target=target)
        out.backward()

        gradients = gradCam.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        activations = gradCam.get_activations(img).detach()

        for j in range(128):  # output filter
            activations[:, j, :, :] *= pooled_gradients[j]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # normalize the heatmap
        min_val = torch.min(heatmap)
        max_val = torch.max(heatmap)

        heatmap = (heatmap - min_val) / (max_val - min_val + 1e-8)

        plot(img, heatmap, i)


if __name__ == "__main__":
    # call json
    model_dir = "log/eval_log/model_for_eval/"
    with open(model_dir + "config.json", "r") as json_file:
        config = json.load(json_file)
    args = DotDict(config)
    args.grid_size = 13
    args.device = torch.device("cpu")

    # call sf
    args.import_sf_model = True
    sf_network = call_sfNetwork(args)
    gradCam = GradCam(sf_network=sf_network, algo_name=args.algo_name)
    target = "s"
    print(f"target Algorithm: {args.algo_name} | target: {target}")

    # call env
    env = get_env(args)
    grid, pos = get_grid(env, args)

    run_loop(grid, pos, target=target)
