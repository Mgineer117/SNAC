import torch
import torch.nn as nn
import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from models.evaulators.base_evaluator import DotDict

from utils import *
from utils.call_env import call_env

import wandb

wandb.require("core")


def remove_dir(dir_path):
    # Iterate over all the files and subdirectories
    for root, dirs, files in os.walk(dir_path, topdown=False):
        # Delete all files
        for name in files:
            os.remove(os.path.join(root, name))
        # Delete all directories
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    # Finally, delete the main directory
    os.rmdir(dir_path)


def run_loop(env_name, option_vals, options):
    # for i in [0, 4, 5, 6, 9]:
    if env_name == "FourRooms" or env_name == "LavaRooms":
        for i in range(10):
            grid, pos, loc = get_grid_tensor(env, env_seed=i)
            save_path = f"RewardMap/{str(i)}"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            else:
                remove_dir(save_path)
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
    elif env_name == "CtF1v1" or env_name == "CtF1v2":
        for i in range(10):
            grid, pos, loc = get_grid_tensor2(env, env_seed=i)
            save_path = f"RewardMap/{str(i)}"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            else:
                remove_dir(save_path)
                os.mkdir(save_path)
            # do reward Plot
            plotter.plotRewardMap2(
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
    args.grid_size = 12
    args.num_vector = 8
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
    env = call_env(args)
    save_dim_to_args(env, args)  # given env, save its state and action dim

    sampler = OnlineSampler(
        training_envs=env,
        state_dim=args.s_dim,
        feature_dim=args.sf_dim,
        action_dim=args.a_dim,
        min_option_length=args.min_option_length,
        min_cover_option_length=args.min_cover_option_length,
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

    run_loop(args.env_name, option_vals, options)
