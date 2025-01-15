import pickle
import wandb
import torch
import json
import os
import sys

sys.path.append("../SNAC")

import gymnasium as gym
from gym_multigrid.envs.fourrooms import FourRooms
from models.evaulators.hc_evaluator import HC_Evaluator
from models.evaulators.base_evaluator import DotDict
from utils.call_network import call_sfNetwork, call_opNetwork, call_hcNetwork
from utils.utils import seed_all
from utils.call_env import call_env
from utils.plotter import Plotter
from torch.utils.tensorboard import SummaryWriter
from log.base_logger import BaseLogger

wandb.require("core")


def train(eval_ep_num=10):
    #### params ####
    env_name = "CtF1v2"
    algo_name = "SNAC+++"
    seed = 1
    seed_all(seed)

    # call configs
    model_dir = f"log/eval_log/model_for_eval/{env_name}/{algo_name}/"
    with open(model_dir + "config.json", "r") as json_file:
        config = json.load(json_file)
    args = DotDict(config)
    args.device = torch.device("cpu")

    # call env
    env = call_env(args)

    # Call loggers
    logdir = "log/eval_log/result/"
    logger = BaseLogger(logdir, name=args.name)
    tensorboard_path = os.path.join(logger.log_dir, "tensorboard")
    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)
    writer = SummaryWriter(log_dir=tensorboard_path)

    sf_path = logger.checkpoint_dirs[0]
    op_path = logger.checkpoint_dirs[1]
    hc_path = logger.checkpoint_dirs[2]

    plotter = Plotter(
        grid_size=args.grid_size,
        img_tile_size=args.img_tile_size,
        hc_path=hc_path,
        log_dir=logger.log_dir,
        device=args.device,
    )

    # import pre-trained model before defining actual models
    print("Loading previous model parameters....")

    args.import_sf_model = True
    args.import_op_model = True
    args.import_hc_model = True

    sf_network = call_sfNetwork(args, sf_path)
    op_network = call_opNetwork(sf_network, args)
    policy = call_hcNetwork(sf_network, op_network, args)

    print(f"Saving Directory = {logdir + args.name}")
    print(f"Result is an average of {eval_ep_num} episodes")

    evaluator_params = {
        "logger": logger,
        "writer": writer,
        "training_env": env,
        "plotter": plotter,
        "gridPlot": True,
        "renderPlot": True,
        "render_fps": 5,
        "gamma": args.gamma,
        "eval_ep_num": 1,
        "episode_len": args.episode_len,
    }

    hc_evaluator = HC_Evaluator(
        dir=hc_path,
        log_interval=1,
        **evaluator_params,
    )

    print("Eval has begun...")
    for epoch in range(eval_ep_num):
        hc_evaluator(policy, epoch=epoch)
        print(f"Eval: {epoch}")

    print("Evaluation is done!")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
