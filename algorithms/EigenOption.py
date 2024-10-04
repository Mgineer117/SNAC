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
    OP_Evaluator,
    UG_Evaluator,
    HC_Evaluator,
)
from models import SFTrainer, OPTrainer, HCTrainer
from utils import *


class EigenOption:
    """
    The difference from SNAC is two-fold:
        - it does not have reward-predictive feature
        - it does only pick top n number of eigenvectors
            - this heuristics ignores other info of eigs
    """

    def __init__(self, env: gym.Env, buffer, sampler, logger, writer, args):
        # object initialization
        self.env = env
        self.buffer = buffer
        self.sampler = sampler
        self.logger = logger
        self.writer = writer
        self.args = args

        # param initialization
        self.curr_epoch = 0

        # SF checkpoint b/c plotter will only be used
        self.sf_path, self.ppo_path, self.op_path, self.ug_path, self.hc_path = (
            self.logger.checkpoint_dirs
        )

        self.plotter = Plotter(
            grid_size=args.grid_size,
            img_tile_size=args.img_tile_size,
            sf_path=self.sf_path,
            ppo_path=self.ppo_path,
            op_path=self.op_path,
            hc_path=self.hc_path,
            log_dir=logger.log_dir,
            device=args.device,
        )

        ### Define evaulators tailored for each process
        # each evaluator has slight deviations
        self.sf_evaluator = SF_Evaluator(
            logger=logger,
            writer=writer,
            training_env=env,
            plotter=self.plotter,
            dir=self.sf_path,
            log_interval=args.log_interval,
        )
        self.op_evaluator = OP_Evaluator(
            logger=logger,
            writer=writer,
            training_env=env,
            plotter=self.plotter,
            dir=self.op_path,
            log_interval=args.log_interval,
            eval_ep_num=5,
        )
        self.ug_evaluator = UG_Evaluator(
            logger=logger,
            writer=writer,
            training_env=env,
            plotter=self.plotter,
            dir=self.ug_path,
            log_interval=args.log_interval,
        )
        self.hc_evaluator = HC_Evaluator(
            logger=logger,
            writer=writer,
            training_env=env,
            plotter=self.plotter,
            dir=self.hc_path,
            log_interval=args.log_interval,
            eval_ep_num=5,
        )

    def run(self):
        self.train_sf()
        self.train_op()
        self.train_hc()

    def train_sf(self):
        ### Call network param and run
        self.sf_network = call_sfNetwork(self.args)
        print_model_summary(self.sf_network, model_name="SF model")
        if not self.args.import_sf_model:
            sf_trainer = SFTrainer(
                policy=self.sf_network,
                sampler=self.sampler,
                buffer=self.buffer,
                logger=self.logger,
                writer=self.writer,
                evaluator=self.sf_evaluator,
                epoch=self.args.SF_epoch,
                init_epoch=self.curr_epoch,
                psi_epoch=self.args.Psi_epoch,
                step_per_epoch=self.args.step_per_epoch,
                eval_episodes=self.args.eval_episodes,
                log_interval=self.args.log_interval,
                env_seed=self.args.env_seed,
            )
            final_epoch = sf_trainer.train()
        else:
            final_epoch = self.curr_epoch + self.args.SF_epoch + self.args.Psi_epoch

        self.curr_epoch += final_epoch

    def train_op(self):
        self.option_vals, self.options, _ = get_eigenvectors(
            self.env,
            self.sf_network,
            self.sampler,
            self.plotter,
            self.args,
            draw_map=True,
        )

        self.op_network = call_opNetwork(
            self.sf_network, self.option_vals, self.options, self.args
        )
        print_model_summary(self.op_network, model_name="OP model")
        if not self.args.import_op_model:
            op_trainer = OPTrainer(
                policy=self.op_network,
                sampler=self.sampler,
                logger=self.logger,
                writer=self.writer,
                evaluator=self.op_evaluator,
                val_options=self.op_network._option_vals,
                epoch=self.curr_epoch + self.args.OP_epoch,
                init_epoch=self.curr_epoch,
                psi_epoch=self.args.Psi_epoch,
                step_per_epoch=self.args.step_per_epoch,
                eval_episodes=self.args.eval_episodes,
                log_interval=self.args.log_interval,
                env_seed=self.args.env_seed,
            )
            final_epoch = op_trainer.train()
        else:
            final_epoch = self.curr_epoch + self.args.OP_epoch + self.args.Psi_epoch
        self.curr_epoch += final_epoch

    def train_hc(self):
        self.hc_network = call_hcNetwork(
            self.sf_network.feaNet, self.op_network, self.args
        )
        print_model_summary(self.hc_network, model_name="HC model")
        if not self.args.import_hc_model:
            hc_trainer = HCTrainer(
                policy=self.hc_network,
                sampler=self.sampler,
                logger=self.logger,
                writer=self.writer,
                evaluator=self.hc_evaluator,
                prefix="HC",
                epoch=self.curr_epoch + self.args.HC_epoch,
                init_epoch=self.curr_epoch,
                step_per_epoch=self.args.step_per_epoch,
                eval_episodes=self.args.eval_episodes,
                log_interval=self.args.log_interval,
                env_seed=self.args.env_seed,
            )
            hc_trainer.train()
        self.curr_epoch += self.args.HC_epoch
