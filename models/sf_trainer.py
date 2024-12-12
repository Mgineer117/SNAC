import time
import random
import os
import cv2
import wandb
import pickle
import numpy as np
from copy import deepcopy
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as multiprocessing

from typing import Optional, Dict, List
from tqdm.auto import trange
from collections import deque
from log.wandb_logger import WandbLogger
from models.policy.base_policy import BasePolicy
from utils.sampler import OnlineSampler
from utils.buffer import TrajectoryBuffer
from models.evaulators.sf_evaluator import Evaluator


# model-free policy trainer
class SFTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        sampler: OnlineSampler,
        buffer: TrajectoryBuffer,
        logger: WandbLogger,
        writer: SummaryWriter,
        evaluator: Evaluator,
        scheduler: torch.optim.lr_scheduler,
        ### Parmaterers ###
        epoch: int = 1000,
        init_epoch: int = 0,
        psi_epoch: int = 20,
        step_per_epoch: int = 1000,
        eval_episodes: int = 10,
        log_interval: int = 2,
        grid_type: int = 0,
    ) -> None:
        self.policy = policy
        self.sampler = sampler
        self.buffer = buffer
        self.evaluator = evaluator

        self.logger = logger
        self.writer = writer

        self.scheduler = scheduler

        # training parameters
        self._init_epoch = init_epoch
        self._epoch = epoch
        self._psi_epoch = psi_epoch
        self._step_per_epoch = step_per_epoch

        self._eval_episodes = eval_episodes

        # initialize the essential training components
        self.last_max_reward = -1e10
        self.last_min_std = 0.0
        self.num_env_steps = 0

        self.log_interval = log_interval
        self.grid_type = grid_type

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        # train loop
        sample_time = self.warm_buffer()

        first_init_epoch = self._init_epoch
        first_final_epoch = self._epoch

        for e in trange(first_init_epoch, first_final_epoch, desc=f"SF Phi Epoch"):
            ### training loop
            self.policy.train()
            for it in trange(self._step_per_epoch, desc=f"Training", leave=False):
                loss, update_time = self.policy.learn(self.buffer)

                loss["SF/sample_time"] = sample_time
                loss["SF/update_time"] = update_time
                loss["SF/lr"] = self.policy.feature_optims.param_groups[0]["lr"]

                self.write_log(loss, iter_idx=int(e * self._step_per_epoch + it))
                sample_time = 0

            if self.scheduler is not None:
                self.scheduler.step()

            if not self.buffer.full:
                batch, sample_time = self.sampler.collect_samples(
                    self.policy, grid_type=self.grid_type
                )
                self.buffer.push(batch)

            ### Eval
            self.policy.eval()  # policy only has to be train_mode in policy_learn, since sampling needs eval_mode as well.
            self.evaluator(
                self.policy,
                epoch=e,
                iter_idx=int(e * self._step_per_epoch + self._step_per_epoch),
                dir_name="SF",
                grid_type=self.grid_type,
            )
            self.save_model(e + 1)
            torch.cuda.empty_cache()

        self.buffer.wipe()

        second_init_epoch = self._epoch
        second_final_epoch = self._epoch + self._psi_epoch
        for e in trange(
            second_init_epoch,
            second_final_epoch,
            desc=f"SF Psi Epoch",
        ):
            self.save_model(e + 1)
            self.policy.train()
            for it in trange(self._step_per_epoch, desc=f"Training", leave=False):
                batch, sample_time = self.sampler.collect_samples(
                    self.policy, grid_type=self.grid_type
                )

                loss, update_time = self.policy.learnPsi(batch)

                loss["SF/sample_time"] = sample_time
                loss["SF/update_time"] = update_time
                loss["SF/train_rew_mean"] = np.mean(batch["rewards"])

                self.write_log(loss, iter_idx=int(e * self._step_per_epoch + it))

            torch.cuda.empty_cache()

        self.policy.eval()
        self.logger.print(
            "total SF training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

        return second_final_epoch

    def save_model(self, e):
        # save checkpoint
        if e % self.log_interval == 0:
            self.policy.save_model(self.logger.checkpoint_dirs[0], e)

    def warm_buffer(self):
        t0 = time.time()
        # make sure there is nothing there
        self.buffer.wipe()

        # collect enough batch
        count = 0
        total_sample_time = 0
        sample_time = 0
        while self.buffer.num_trj < self.buffer.min_num_trj:
            batch, sampleT = self.sampler.collect_samples(
                self.policy, grid_type=self.grid_type
            )
            self.buffer.push(batch, post_process="nonzero_rewards")
            sample_time += sampleT
            total_sample_time += sampleT
            if count % 100 == 0:
                print(
                    f"\nWarming buffer with nonzero reward {self.buffer.num_trj}/{self.buffer.min_num_trj} | sample_time = {sample_time:.2f}s",
                    end="",
                )
                sample_time = 0
            count += 1
        print(
            f"\nWarming Complete! {self.buffer.num_trj}/{self.buffer.min_num_trj} | total sample_time = {total_sample_time:.2f}s",
            end="",
        )
        print()
        t1 = time.time()
        sample_time = t1 - t0
        return sample_time

    def write_log(self, logging_dict: dict, iter_idx: int):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(int(iter_idx), display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, int(iter_idx))
