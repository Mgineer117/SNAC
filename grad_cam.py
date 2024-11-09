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
            reward = self.multiply_options(x_r, self.options)
            reward = reward[0][0].detach().numpy()
            x = torch.sum(x_s, dim=-1)
            if target == "s":
                x = torch.sum(x_s, dim=-1)
            elif target == "r":
                x = torch.sum(x_r, dim=-1)
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
    if env_name == 'FourRooms' or env_name == 'LavaRooms':
        grid, (x_coords, y_coords), loc = get_grid_tensor(env, args.grid_type)
    else:
        grid, (x_coords, y_coords), loc = get_grid_tensor2(env, args.grid_type)

    grid = grid[None, :, :, :]
    pos = np.hstack((x_coords[:, None], y_coords[:, None]))

    grid = torch.from_numpy(grid).to(torch.float32)
    pos = torch.from_numpy(pos).to(torch.float32)
    return grid, pos


def plot(img, heatmap, reward, i):
    img = img.numpy()
    heatmap = heatmap.numpy()

    colors = [
            (0.2, 0.2, 1),
            (0.2667, 0.0039, 0.3294),
            (1, 0.2, 0.2),
        ]  # Blue -> Black -> Red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pale_blue_dark_pale_red", colors
    )

    # Figure setup
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the original image
    axes[0].imshow(np.flipud(img), cmap='viridis')
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Plot the heatmap with custom colormap
    im = axes[1].imshow(np.flipud(heatmap), cmap=cmap)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], orientation='vertical')

    # Resize heatmap to match img size and overlay
    alpha = 0.9
    resized_heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
    superimposed_img = resized_heatmap * alpha + img * (1 - alpha)

    # Plot the superimposed image
    axes[2].imshow(np.flipud(superimposed_img), cmap=cmap)
    axes[2].set_title(f"Super-imposed with reward: {reward:.1f}")
    axes[2].axis("off")

    # Save and close
    plt.savefig(f"heatmap/{i}.png", bbox_inches='tight')
    plt.close()


def run_loop(env_name, grid, pos, target="s"):
    for i in range(pos.shape[0]):
        x, y = pos[i, 0], pos[i, 1]

        img = grid.clone()
        
        if env_name == 'FourRooms' or env_name == 'LavaRooms':
            img[:, x.long(), y.long(), 0] = 10
        else:
            img[:, x.long(), y.long(), 1] = 1
            img[:, x.long(), y.long(), 2] = 2
        # do grad-cam
        out, reward = gradCam(img, pos[i, :].unsqueeze(0), target=target)
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
        heatmap = 2*(heatmap - min_val) / (max_val - min_val + 1e-8) - 1

        img = torch.sum(img[0, :, :, :], axis=-1)
        min_val = torch.min(img)
        max_val = torch.max(img)
        img = 2*(img - min_val) / (max_val - min_val + 1e-8) - 1
        plot(img, heatmap, reward, i)


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
    target = "r"
    print(f"target Algorithm: {args.algo_name} | target: {target}")

    # call env
    env = call_env(args)
    grid, pos = get_grid(env, args.env_name, args)

    run_loop(args.env_name, grid, pos, target=target)
