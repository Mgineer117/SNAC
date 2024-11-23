import numpy as np
import torch
import torch.nn as nn
import random

from gym_multigrid.envs.fourrooms import FourRooms
from gym_multigrid.envs.lavarooms import LavaRooms
from gym_multigrid.envs.ctf import CtF
import safety_gymnasium as sgym
import gymnasium as gym

from utils.wrappers import GridWrapper, CtFWrapper, NavigationWrapper


def disc_or_cont(env, args):
    if isinstance(env.action_space, gym.spaces.Discrete):
        args.is_discrete = True
    elif isinstance(env.action_space, gym.spaces.Box):
        args.is_discrete = False
    else:
        raise ValueError(f"Unknown action space type {env.action_space}.")


def call_env(args):
    # define the env
    if args.env_name == "FourRooms":
        # first call dummy env to find possible location for agent
        env = FourRooms(
            grid_type=args.grid_type,
            max_steps=args.episode_len,
            tile_size=args.img_tile_size,
            highlight_visible_cells=False,
            partial_observability=False,
            render_mode="rgb_array",
        )
        disc_or_cont(env, args)
        return GridWrapper(env, tile_size=args.tile_size)
    elif args.env_name == "LavaRooms":
        # first call dummy env to find possible location for agent
        env = LavaRooms(
            grid_type=args.grid_type,
            max_steps=args.episode_len,
            tile_size=args.img_tile_size,
            highlight_visible_cells=False,
            partial_observability=False,
            render_mode="rgb_array",
        )
        disc_or_cont(env, args)
        return GridWrapper(env, tile_size=args.tile_size)

    elif args.env_name in ("CtF1v1", "CtF1v2", "CtF1v3", "CtF1v4"):
        map_path: str = "assets/ctf_avoid_obj.txt"
        observation_option: str = "tensor"
        env_name = args.env_name
        red_agents = int(env_name.split('v')[1])  # Extract the number of red agents from the env name
        if env_name.startswith("CtF1v"):
            env = CtF(
                map_path=map_path,
                num_blue_agents=1,
                num_red_agents=red_agents,
                observation_option=observation_option,
                step_penalty_ratio=0.0,
            )
        else:
            raise NotImplementedError(f"{args.env_name} not implemented")
        disc_or_cont(env, args)
        return CtFWrapper(env, tile_size=args.tile_size)
    elif args.env_name == "PointNavigation":
        env = sgym.make(
            "SafetyPointButton1-v0",
            render_mode="rgb_array",
            max_episode_steps=args.episode_len,
            width=512,
            height=512,
            camera_name="fixedfar"
        )
        disc_or_cont(env, args)
        return NavigationWrapper(
            env, tile_size=args.tile_size, cost_scaler=args.cost_scaler
        )
    else:
        raise ValueError(f"Invalid environment key: {args.env_name}")
