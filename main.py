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

from utils import *
from utils.call_env import call_env

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
    env = call_env(args)
    save_dim_to_args(env, args)  # given env, save its state and action dim
    logger, writer = setup_logger(args, unique_id, seed)

    if args.algo_name == "SNAC":
        # start the sf training or import it
        ft = FeatureTrain(env=env, logger=logger, writer=writer, args=args)
        sf_network, prev_epoch = ft.train()
        alg = SNAC(
            env=env,
            sf_network=sf_network,
            prev_epoch=prev_epoch,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif (
        args.algo_name == "EigenOption"
        or args.algo_name == "EigenOption2"
        or args.algo_name == "EigenOption3"
    ):
        # start the sf training or import it
        ft = FeatureTrain(env=env, logger=logger, writer=writer, args=args)
        sf_network, prev_epoch = ft.train()
        alg = EigenOption(
            env=env,
            sf_network=sf_network,
            prev_epoch=prev_epoch,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "CoveringOption":
        # start the sf training or import it
        ft = FeatureTrain(env=env, logger=logger, writer=writer, args=args)
        sf_network, prev_epoch = ft.train()
        alg = CoveringOption(
            env=env,
            sf_network=sf_network,
            prev_epoch=prev_epoch,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "PPO":
        alg = PPO(
            env=env,
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
