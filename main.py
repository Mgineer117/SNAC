import torch
import torch.nn as nn
import random
import uuid
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from algorithms.SNAC import SNAC
from algorithms.EigenOption import EigenOption
from algorithms.CoveringOption import CoveringOption
from algorithms.PPO import PPO
from algorithms.FeatureTrain import FeatureTrain
from gym_multigrid.envs.fourrooms import FourRooms

from utils import *

import wandb

wandb.require("core")


def check_all_devices(module):
    devices = {param.device for param in module.parameters()}  # Get all unique devices
    return devices


#########################################################
# Parameter definitions
#########################################################
def train(args, unique_id):
    # call logger
    logger, writer = setup_logger(args, unique_id, seed)

    # start the sf training or import it
    ft = FeatureTrain(logger=logger, writer=writer, args=args)
    sf_network, prev_epoch = ft.train()

    if args.algo_name == "SNAC":
        alg = SNAC(
            sf_network=sf_network,
            prev_epoch=prev_epoch,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "EigenOption":
        alg = EigenOption(
            sf_network=sf_network,
            prev_epoch=prev_epoch,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "CoveringOption":
        alg = CoveringOption(
            sf_network=sf_network,
            prev_epoch=prev_epoch,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "PPO":
        alg = PPO(
            logger=logger,
            writer=writer,
            args=args,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algo_name}")

    alg.run()

    wandb.finish()
    writer.close()


#########################################################
# ENV LOOP
#########################################################

if __name__ == "__main__":
    # initialize for whole training pipeline
    args = get_args(verbose=False)
    seeds = args.seeds
    # define unique id for the run
    unique_id = str(uuid.uuid4())[:4]
    # iterate over seeds
    for seed in seeds:
        args = get_args()
        seed_all(seed)
        train(args, unique_id)
