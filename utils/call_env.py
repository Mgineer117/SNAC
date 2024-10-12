import numpy as np
import torch
import torch.nn as nn
import random

from gym_multigrid.envs.fourrooms import FourRooms
from gym_multigrid.envs.ctf import Ctf1v1Env
from gym_multigrid.policy.ctf.heuristic import 
from utils import NoStateDictWrapper, get_grid_tensor
from utils.wrappers import NoStateDictCtfWrapper


def call_env(args, fix_agent_pos: bool = True):
    # define the env
    if args.env_name == "FourRooms":
        # first call dummy env to find possible location for agent
        args.grid_size = 13  # Machado reference
        args.a_dim = 4
        if fix_agent_pos:
            dummy_env = FourRooms(
                grid_size=(args.grid_size, args.grid_size),
                agent_pos=(5, 5),
                max_steps=args.episode_len,
                tile_size=args.img_tile_size,
                highlight_visible_cells=False,
                partial_observability=False,
                render_mode="rgb_array",
            )
            dummy_env = NoStateDictWrapper(dummy_env, tile_size=args.tile_size)

            # given possible locations of agent placement, fix it according to env_seed
            _, coords, _ = get_grid_tensor(dummy_env, args.env_seed)
            x_coord, y_coord = random.choice(coords[0]), random.choice(coords[1])

            env = FourRooms(
                grid_size=(args.grid_size, args.grid_size),
                agent_pos=(x_coord, y_coord),
                max_steps=args.episode_len,
                tile_size=args.img_tile_size,
                highlight_visible_cells=False,
                partial_observability=False,
                render_mode="rgb_array",
            )
        else:
            env = FourRooms(
                grid_size=(args.grid_size, args.grid_size),
                max_steps=args.episode_len,
                tile_size=args.img_tile_size,
                highlight_visible_cells=False,
                partial_observability=False,
                render_mode="rgb_array",
            )
        env = NoStateDictWrapper(env, tile_size=args.tile_size)

    elif args.env_name == "CtF1v1" or "CtF1v2":
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
