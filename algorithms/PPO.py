import torch
import torch.nn as nn
import random
import uuid
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gym_multigrid.envs.fourrooms import FourRooms

from models.evaulators import SF_Evaluator, PPO_Evaluator
from models import SFTrainer, PPOTrainer
from utils import *


class PPO:
    def __init__(self, env: gym.Env, buffer, sampler, logger, writer, args):
        """
        This is a naive PPO wrapper that includes all necessary training pipelines for HRL.
        This trains SF network and train PPO according to the extracted features by SF network
        """
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
        self.ppo_evaluator = PPO_Evaluator(
            logger=logger,
            writer=writer,
            training_env=env,
            plotter=self.plotter,
            dir=self.op_path,
            log_interval=args.log_interval,
            eval_ep_num=10,
        )

    def run(self):
        self.train_sf()
        self.train_ppo()

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

    def train_ppo(self):
        ### Call network param and run
        self.ppo_network = call_ppoNetwork(self.sf_network, self.args)
        print_model_summary(self.ppo_network, model_name="PPO model")
        if not self.args.import_ppo_model:
            ppo_trainer = PPOTrainer(
                policy=self.ppo_network,
                sampler=self.sampler,
                logger=self.logger,
                writer=self.writer,
                evaluator=self.ppo_evaluator,
                epoch=self.curr_epoch + self.args.PPO_epoch,
                init_epoch=self.curr_epoch,
                step_per_epoch=self.args.step_per_epoch,
                eval_episodes=self.args.eval_episodes,
                log_interval=self.args.log_interval,
                env_seed=self.args.env_seed,
            )
            final_epoch = ppo_trainer.train()
        else:
            final_epoch = self.curr_epoch + self.args.PPO_epoch

        self.curr_epoch += final_epoch
