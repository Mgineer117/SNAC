import torch
import torch.nn as nn
import random
import uuid
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gym_multigrid.envs.fourrooms import FourRooms

from models.evaulators import (
    SF_Evaluator,
    PPO_Evaluator,
    OP_Evaluator,
    UG_Evaluator,
    HC_Evaluator,
)
from models import SFTrainer, PPOTrainer, OPTrainer, UGComparer, HCTrainer

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

    ### define essential componenets for project
    ### env here must be grid & agent fixed while others stochastic
    # get the possible coordinates for agent allocation
    _, coords, _ = get_grid_tensor(env, args.env_seed)
    x_coord, y_coord = random.choice(coords[0]), random.choice(coords[1])

    # define the env
    args.grid_size = 13  # Machado reference
    env = FourRooms(
        grid_size=(args.grid_size, args.grid_size),
        agent_pos=(x_coord, y_coord),
        max_steps=args.episode_len,
        img_tile_size=args.img_tile_size,
        highlight=False,
        partial_observability=False,
        render_mode="rgb_array",
    )
    env = NoStateDictWrapper(env, tile_size=args.tile_size)
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

    ### Prepare training modules
    sf_path, ppo_path, op_path, ug_path, hc_path = (
        logger.checkpoint_dirs
    )  # SF checkpoint b/c plotter will only be used

    plotter = Plotter(
        grid_size=args.grid_size,
        img_tile_size=args.img_tile_size,
        sf_path=sf_path,
        ppo_path=ppo_path,
        op_path=op_path,
        hc_path=hc_path,
        log_dir=logger.log_dir,
        device=args.device,
    )

    ### Define evaulators tailored for each process
    # each evaluator has slight deviations
    SF_evaluator = SF_Evaluator(
        logger=logger,
        writer=writer,
        training_env=env,
        plotter=plotter,
        dir=sf_path,
        log_interval=args.log_interval,
    )
    OP_evaluator = OP_Evaluator(
        logger=logger,
        writer=writer,
        training_env=env,
        plotter=plotter,
        dir=op_path,
        log_interval=args.log_interval,
        eval_ep_num=5,
    )
    UG_evaluator = UG_Evaluator(
        logger=logger,
        writer=writer,
        training_env=env,
        plotter=plotter,
        dir=ug_path,
        log_interval=args.log_interval,
    )
    HC_evaluator = HC_Evaluator(
        logger=logger,
        writer=writer,
        training_env=env,
        plotter=plotter,
        dir=hc_path,
        log_interval=args.log_interval,
        eval_ep_num=5,
    )

    ### Call network param and run
    curr_epoch = 0
    sf_network = call_sfNetwork(args)
    print_model_summary(sf_network, model_name="SF model")
    if not args.import_sf_model:
        sf_trainer = SFTrainer(
            policy=sf_network,
            sampler=sampler,
            buffer=buffer,
            logger=logger,
            writer=writer,
            evaluator=SF_evaluator,
            epoch=args.SF_epoch,
            init_epoch=curr_epoch,
            psi_epoch=args.Psi_epoch,
            step_per_epoch=args.step_per_epoch,
            eval_episodes=args.eval_episodes,
            log_interval=args.log_interval,
            env_seed=args.env_seed,
        )
        final_epoch = sf_trainer.train()
    else:
        final_epoch = curr_epoch + args.SF_epoch + args.Psi_epoch

    curr_epoch += final_epoch

    ### Discover option set
    option_vals, options = get_eigenvectors(
        env, sf_network, sampler, plotter, args, draw_map=True
    )
    # random psi and option psi wrappers
    op_network = call_opNetwork(sf_network, option_vals, options, args)
    print_model_summary(op_network, model_name="OP model")
    if not args.import_op_model:
        op_trainer = OPTrainer(
            policy=op_network,
            sampler=sampler,
            logger=logger,
            writer=writer,
            evaluator=OP_evaluator,
            val_options=op_network._option_vals,
            epoch=curr_epoch + args.OP_epoch,
            init_epoch=curr_epoch,
            psi_epoch=args.Psi_epoch,
            step_per_epoch=args.step_per_epoch,
            eval_episodes=args.eval_episodes,
            log_interval=args.log_interval,
            env_seed=args.env_seed,
        )
        final_epoch = op_trainer.train()
    else:
        final_epoch = curr_epoch + args.OP_epoch + args.Psi_epoch
    curr_epoch += final_epoch

    # ### Create a actionValue Map
    # for z in range(eigs[-1]):
    #     plotter.plotActionValueMap2(
    #         feaNet=sf_network.feaNet,
    #         psiNet=op_network.psiNet,
    #         S=eigs[0][z].unsqueeze(0),
    #         V=eigs[1][z, :].unsqueeze(0),
    #         z=z,
    #         grid_tensor=grids[0],
    #         coords=grids[1],
    #         loc=grids[2],
    #         specific_path=str(z),
    #     )

    rp_network = call_rpNetwork(
        sf_network.feaNet, sf_network.psiNet, op_network._options, args
    )
    print_model_summary(rp_network, model_name="RC model")

    ug_comparer = UGComparer(
        uniform_policy=rp_network,
        option_policy=op_network,
        sampler=sampler,
        buffer=buffer,
        logger=logger,
        writer=writer,
        evaluator=UG_evaluator,
        init_epoch=curr_epoch,
        step_per_epoch=args.step_per_epoch,
        env_seed=args.env_seed,
    )
    ug_comparer.train()
    curr_epoch += op_network._num_options

    hc_network = call_hcNetwork(sf_network.feaNet, op_network, args)
    print_model_summary(hc_network, model_name="HC model")
    if not args.import_hc_model:
        hc_trainer = HCTrainer(
            policy=hc_network,
            sampler=sampler,
            logger=logger,
            writer=writer,
            evaluator=HC_evaluator,
            prefix="HC_OP",
            epoch=curr_epoch + args.HC_epoch,
            init_epoch=curr_epoch,
            step_per_epoch=args.step_per_epoch,
            eval_episodes=args.eval_episodes,
            log_interval=args.log_interval,
            env_seed=args.env_seed,
        )
        hc_trainer.train()
    curr_epoch += args.HC_epoch

    hc_network = call_hcNetwork(sf_network.feaNet, rp_network, args)
    if not args.import_hc_model:
        hc_trainer = HCTrainer(
            policy=hc_network,
            sampler=sampler,
            logger=logger,
            writer=writer,
            evaluator=HC_evaluator,
            prefix="HC_RP",
            epoch=curr_epoch + args.HC_epoch,
            init_epoch=curr_epoch,
            step_per_epoch=args.step_per_epoch,
            eval_episodes=args.eval_episodes,
            log_interval=args.log_interval,
            env_seed=args.env_seed,
        )
        hc_trainer.train()
    curr_epoch += args.HC_epoch

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
        seed_all(seed)
        train(args, unique_id)
