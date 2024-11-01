import numpy as np
import torch
import torch.nn as nn
import random

from gym_multigrid.envs.fourrooms import FourRooms
from gym_multigrid.envs.lavarooms import LavaRooms

from gym_multigrid.envs.ctf import CtF
from utils import NoStateDictWrapper, get_grid_tensor
from utils.wrappers import NoStateDictCtfWrapper


def call_env(args):
    # define the env
    if args.env_name == "FourRooms":
        # first call dummy env to find possible location for agent
        env = FourRooms(
            env_seed=args.env_seed,
            grid_size=(13, 13),  # fixed
            max_steps=args.episode_len,
            tile_size=args.img_tile_size,
            highlight_visible_cells=False,
            partial_observability=False,
            render_mode="rgb_array",
        )
        return NoStateDictWrapper(env, tile_size=args.tile_size)
    elif args.env_name == "LavaRooms":
        # first call dummy env to find possible location for agent
        env = LavaRooms(
            env_seed=args.env_seed,
            grid_size=(9, 9),  # fixed
            max_steps=args.episode_len,
            tile_size=args.img_tile_size,
            highlight_visible_cells=False,
            partial_observability=False,
            render_mode="rgb_array",
        )
        return NoStateDictWrapper(env, tile_size=args.tile_size)

    elif args.env_name == "CtF1v1" or "CtF1v2" or "CtF1v3" or "CtF1v4":
        args.a_dim = 5
        map_path: str = "assets/ctf_avoid_obj.txt"
        observation_option: str = "tensor"
        if args.env_name == "CtF1v1":
            env = CtF(
                map_path=map_path,
                num_blue_agents=1,
                num_red_agents=1,
                observation_option=observation_option,
                step_penalty_ratio=0.0,
            )
        elif args.env_name == "CtF1v2":
            env = CtF(
                map_path=map_path,
                num_blue_agents=1,
                num_red_agents=2,
                observation_option=observation_option,
                step_penalty_ratio=0.0,
            )
        elif args.env_name == "CtF1v3":
            env = CtF(
                map_path=map_path,
                num_blue_agents=1,
                num_red_agents=3,
                observation_option=observation_option,
                step_penalty_ratio=0.0,
            )
        elif args.env_name == "CtF1v4":
            env = CtF(
                map_path=map_path,
                num_blue_agents=1,
                num_red_agents=4,
                observation_option=observation_option,
                step_penalty_ratio=0.0,
            )
        else:
            raise NotImplementedError(f"{args.env_name} not implemented")
        return NoStateDictCtfWrapper(env, tile_size=args.tile_size)
    else:
        raise ValueError(f"Invalid environment key: {args.env_name}")
