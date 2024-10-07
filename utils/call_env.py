import numpy as np
import torch
import torch.nn as nn
import random

from gym_multigrid.envs.fourrooms import FourRooms
from gym_multigrid.envs.ctf import Ctf1v1Env
from utils import NoStateDictWrapper, get_grid_tensor


def call_env(args):
    # define the env
    # first call dummy env to find possible location for agent
    args.grid_size = 13  # Machado reference
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
    env = NoStateDictWrapper(env, tile_size=args.tile_size)

    return env


def call_env_ctf(args):
    args.grid_size = 12
    args.feat_net_type = "VAE"
    map_path: str = "assets/ctf_avoid_obj.txt"
    observation_option: str = "pos_map_flattened"
    env = Ctf1v1Env(map_path=map_path, observation_option=observation_option)
    return env
