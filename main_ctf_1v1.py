import uuid

from algorithms.SNAC import SNAC
from algorithms.EigenOption import EigenOption
from algorithms.CoveringOption import CoveringOption
from algorithms.PPO import PPO

from utils import *
from utils.call_env import call_env_ctf

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

    ### define essential componenets for project
    ### env here must be grid & agent fixed while others stochastic
    # get the possible coordinates for agent allocation
    env = call_env_ctf(args)
    save_dim_to_args(env, args)  # given env, save its state and action dim

    # define buffers and sampler for Monte-Carlo sampling
    buffer = TrajectoryBuffer(
        min_num_trj=args.update_iter * args.trj_per_iter, max_num_trj=args.num_traj
    )
    sampler = OnlineSampler(
        training_envs=env,
        state_dim=args.s_dim,
        feature_dim=args.sf_dim,
        action_dim=args.a_dim,
        episode_len=args.episode_len,
        episode_num=args.episode_num,
        num_cores=args.num_cores,
    )

    if args.algo_name == "SNAC":
        alg = SNAC(
            env=env,
            buffer=buffer,
            sampler=sampler,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "EigenOption":
        alg = EigenOption(
            env=env,
            buffer=buffer,
            sampler=sampler,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "CoveringOption":
        alg = CoveringOption(
            env=env,
            buffer=buffer,
            sampler=sampler,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "PPO":
        alg = PPO(
            env=env,
            buffer=buffer,
            sampler=sampler,
            logger=logger,
            writer=writer,
            args=args,
        )
    alg.run()

    wandb.finish()
    writer.close()


#########################################################
# ENV LOOP
#########################################################

if __name__ == "__main__":
    args = get_args(verbose=False)
    seeds = args.seeds
    unique_id = str(uuid.uuid4())[:4]
    for seed in seeds:
        args = get_args()
        args.algo_name: str = "SNAC"

        # args.SF_epoch = 1
        # args.OP_epoch = 1
        # args.step_per_epoch = 1
        # args.HC_epoch = 1

        # args.draw_map: bool = True
        seed_all(seed)
        train(args, unique_id)
