import torch
import torch.nn as nn
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gymnasium as gym

from models.evaulators.base_evaluator import DotDict

# from gym_multigrid.envs.fourrooms import FourRooms
from gym_multigrid.envs.fourrooms import FourRooms
from gym_multigrid.envs.lavarooms import LavaRooms

from utils import *
from utils.call_env import call_env

import wandb

wandb.require("core")


class GradCam(nn.Module):
    def __init__(self, sf_network, algo_name):
        super(GradCam, self).__init__()

        # get the pretrained VGG19 network
        self.feaNet = sf_network.feaNet
        self.options = sf_network._options
        self.preGrad = sf_network.feaNet.pre_grad_cam
        self.postGrad = sf_network.feaNet.post_grad_cam

        self.multiply_options = lambda x, y: torch.sum(
            torch.mul(x, y), axis=-1, keepdim=True
        )

        self.algo_name = algo_name

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, pos, target="s"):
        x = self.preGrad(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        x = self.postGrad(x, pos)

        # apply the remaining pooling
        if self.algo_name == "SNAC":
            x_r, x_s = torch.split(x, x.size(-1) // 2, dim=-1)
            x = torch.sum(x_s, dim=-1)
            if target == "s":
                x = torch.sum(x_s, dim=-1)
                reward = 0
            elif target == "r":
                x = torch.sum(x_r, dim=-1)
                reward = self.multiply_options(x_r, self.options)
                reward = reward[0][0].detach().numpy()
            else:
                raise ValueError(f"Unknown target: {target}")
        else:
            reward = 0
            x = torch.sum(x)
        return x, reward

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.preGrad(x)


def get_grid(env, env_name, args):
    grid, (x_coords, y_coords), agent_pos = get_grid_tensor(env, args.grid_type)

    grid = grid[None, :, :, :]
    coords = np.hstack((x_coords[:, None], y_coords[:, None]))

    grid = torch.from_numpy(grid).to(torch.float32)
    coords = torch.from_numpy(coords).to(torch.float32)
    agent_pos = torch.from_numpy(agent_pos).to(torch.float32)
    return grid, coords, agent_pos


def plot(img, heatmap, reward, i):
    img = img.numpy()
    heatmap = heatmap.numpy()

    colors = [
        (0.2, 0.2, 1),
        (0.2667, 0.0039, 0.3294),
        (1, 0.2, 0.2),
    ]  # Blue -> Black -> Red
    cmap = mcolors.LinearSegmentedColormap.from_list("pale_blue_dark_pale_red", colors)

    # Figure setup
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the original image
    axes[0].imshow(np.flipud(img), cmap="viridis")
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Plot the heatmap with custom colormap
    im = axes[1].imshow(np.flipud(heatmap), cmap=cmap)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], orientation="vertical")

    # Resize heatmap to match img size
    resized_heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))

    # Superimpose image and heatmap in third plot
    # Plot `img` as the base with `viridis` colormap
    axes[2].imshow(np.flipud(img), cmap="viridis")

    # Overlay resized `heatmap` using the custom colormap and alpha for blending
    axes[2].imshow(np.flipud(resized_heatmap), cmap=cmap, alpha=0.5)
    axes[2].set_title(f"Super-imposed with reward: {reward:.3f}")
    axes[2].axis("off")

    # Save and close
    plt.savefig(f"heatmap/{i}.png", bbox_inches="tight")
    plt.close()


def run_loop(env_name, grid, coords, agent_pos, target="s"):
    for i in range(coords.shape[0]):
        x, y = coords[i, 0], coords[i, 1]

        img = grid.clone()
        pos = agent_pos.clone()

        if env_name == "FourRooms" or env_name == "LavaRooms":
            img[:, x.long(), y.long(), 0] = 10
        else:
            img[:, x.long(), y.long(), 1] = 1
            img[:, x.long(), y.long(), 2] = 2

        pos[0] = x
        pos[1] = y

        # do grad-cam
        out, reward = gradCam(img, pos.unsqueeze(0), target=target)
        out.backward()

        gradients = gradCam.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = gradCam.get_activations(img).detach()

        for j in range(128):
            activations[:, j, :, :] *= pooled_gradients[j]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # normalize the heatmap
        min_val = torch.min(heatmap)
        max_val = torch.max(heatmap)
        heatmap = 2 * (heatmap - min_val) / (max_val - min_val + 1e-8) - 1

        if env_name == "CtF1v1" or env_name == "CtF1v2":
            # reassign the agent
            obj_indices = img[0, :, :, 1] != 0
            obj = (img[0, :, :, 1] + 1) * 2
            img = torch.sum(img[0, :, :, :], axis=-1)
            img[obj_indices] = obj[obj_indices]
        else:
            img = torch.sum(img[0, :, :, :], axis=-1)
        min_val = torch.min(img)
        max_val = torch.max(img)
        img = 2 * (img - min_val) / (max_val - min_val + 1e-8) - 1
        plot(img, heatmap, reward, i)


if __name__ == "__main__":
    # call json
    json_dir = f"log/eval_log/model_for_eval/"
    with open(json_dir + "config.json", "r") as json_file:
        config = json.load(json_file)
    args = DotDict(config)
    args.env_name = "CtF1v2"
    args.algo_name = "SNAC"
    args.device = torch.device("cpu")

    print(f"Algo name: {args.algo_name}")
    print(f"Env name: {args.env_name}")

    # call sf
    args.import_sf_model = True
    sf_network = call_sfNetwork(args)
    gradCam = GradCam(sf_network=sf_network, algo_name=args.algo_name)
    target = "r"
    print(f"target Algorithm: {args.algo_name} | target: {target}")

    # call env
    env = call_env(args)
    grid, coords, agent_pos = get_grid(env, args.env_name, args)

    run_loop(args.env_name, grid, coords, agent_pos, target=target)
