import torch
import torch.nn as nn
import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from models.evaulators.base_evaluator import DotDict
from gym_multigrid.envs.lavarooms import LavaRooms

from utils import *

import wandb

wandb.require("core")


def get_env(args):
    env = LavaRooms(
        grid_size=(args.grid_size, args.grid_size),
        max_steps=args.episode_len,
        tile_size=args.img_tile_size,
        highlight_visible_cells=False,
        partial_observability=False,
        render_mode="rgb_array",
    )
    env = NoStateDictWrapper(env, tile_size=args.tile_size)
    return env


def run_loop(option_vals, options):
    # for i in [0, 4, 5, 6, 9]:
    for i in range(10):
        grid, pos, loc = get_grid_tensor(env, env_seed=i)
        save_path = f"RewardMap/{str(i)}"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # do reward Plot
        plotter.plotRewardMap(
            feaNet=sf_network.feaNet,
            S=option_vals,
            V=options,
            feature_dim=args.sf_dim,
            algo_name=args.algo_name,
            grid_tensor=grid,
            coords=pos,
            loc=loc,
            dir=save_path,
            device=args.device,
        )


if __name__ == "__main__":
    # call json
    model_dir = "log/eval_log/model_for_eval/"
    with open(model_dir + "config.json", "r") as json_file:
        config = json.load(json_file)
    args = DotDict(config)
    args.grid_size = 9
    args.num_vector = 12
    args.device = torch.device("cpu")

    # call sf
    args.import_sf_model = True
    sf_network = call_sfNetwork(args)
    plotter = Plotter(
        grid_size=args.grid_size,
        img_tile_size=args.img_tile_size,
        device=args.device,
    )

    # call env
    env = get_env(args)
    save_dim_to_args(env, args)  # given env, save its state and action dim

    sampler = OnlineSampler(
        training_envs=env,
        state_dim=args.s_dim,
        feature_dim=args.sf_dim,
        action_dim=args.a_dim,
        episode_len=args.episode_len,
        episode_num=args.episode_num,
        num_cores=args.num_cores,
    )

    option_vals, options, _ = get_eigenvectors(
        env,
        sf_network,
        sampler,
        plotter,
        args,
        draw_map=False,
    )

    print(option_vals.shape, options.shape)

    run_loop(option_vals, options)
