import time
import random
import math
import os
import wandb
import pickle
import numpy as np
from copy import deepcopy
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

from typing import Optional, Dict, List
from tqdm.auto import trange
from collections import deque
from log.wandb_logger import WandbLogger
from models.policy.optionPolicy import OP_Controller
from utils.sampler import OnlineSampler
from utils.buffer import TrajectoryBuffer
from models.evaulators.sf_evaluator import Evaluator


# Custom scheduler logic for different parameter groups
def custom_lr_scheduler(optimizer, epoch, scheduling_epoch=1):
    if epoch % scheduling_epoch == 0:
        optimizer["ppo"].param_groups[0]["lr"] *= 0.7  # Reduce learning rate for phi


# model-free policy trainer
class OPTrainer:
    def __init__(
        self,
        policy: OP_Controller,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        evaluator: Evaluator,
        val_options: torch.Tensor,
        epoch: int = 1000,
        init_epoch: int = 0,
        step_per_epoch: int = 1000,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        log_interval: int = 2,
        grid_type: int = 0,
    ) -> None:
        self.policy = policy
        self.sampler = sampler
        self.evaluator = evaluator

        self.logger = logger
        self.writer = writer

        # training parameters
        self._epoch = epoch
        self._init_epoch = init_epoch
        self._step_per_epoch = step_per_epoch

        self._eval_episodes = eval_episodes
        self._scheduling_epoch = int(self._epoch // 10) if self._epoch >= 10 else None
        self._val_options = val_options
        self.lr_scheduler = lr_scheduler

        # initialize the essential training components
        self.last_max_reward = -1e10
        self.std_limit = 0.5
        self.num_env_steps = 0

        self.log_interval = log_interval
        self.grid_type = grid_type

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        self.last_reward_mean = deque(maxlen=3)
        self.last_reward_std = deque(maxlen=3)

        # train loop
        self.policy.train()
        first_init_epoch = self._init_epoch
        first_final_epoch = self._epoch
        for e in trange(first_init_epoch, first_final_epoch, desc=f"OP PPO Epoch"):
            ### training loop
            for it in trange(self._step_per_epoch, desc=f"Training", leave=False):
                sample_time = 0
                update_time = 0
                policy_loss = []
                avgRewDictList = []
                for z in trange(
                    self.policy._num_options, desc=f"Updating Option", leave=False
                ):
                    # sample batch
                    batch, sampleT = self.sampler.collect_samples(
                        self.policy, idx=z, grid_type=self.grid_type
                    )
                    sample_time += sampleT

                    # update params
                    loss_dict, avgRewDict, updateT = self.policy.learn(batch, z)
                    policy_loss.append(loss_dict)
                    avgRewDictList.append(avgRewDict)
                    update_time += updateT

                loss = self.average_dict_values(policy_loss)

                # Logging further info
                loss["OP/sample_time"] = sample_time
                loss["OP/update_time"] = update_time
                for each_dict in avgRewDictList:
                    loss.update(each_dict)

                self.write_log(loss, iter_idx=int(e * self._step_per_epoch + it))

            ### Eval Loop
            self.policy.eval()
            rew_mean = np.zeros((self.policy._num_options,))
            rew_std = np.zeros((self.policy._num_options,))
            ln_mean = np.zeros((self.policy._num_options,))
            ln_std = np.zeros((self.policy._num_options,))
            for z in trange(self.policy._num_options, desc=f"Evaluation", leave=False):
                eval_dict = self.evaluator(
                    self.policy,
                    epoch=e + 1,
                    iter_idx=int(e * self._step_per_epoch + self._step_per_epoch),
                    idx=z,
                    name1=self._val_options[z],
                    dir_name="OP",
                    write_log=False,  # since OP needs to write log of average of all options
                    grid_type=self.grid_type,
                )
                rew_mean[z] = eval_dict["rew_mean"]
                rew_std[z] = eval_dict["rew_std"]
                ln_mean[z] = eval_dict["ln_mean"]
                ln_std[z] = eval_dict["ln_std"]

            rew_mean = np.mean(rew_mean)
            rew_std = np.mean(rew_std)
            ln_mean = np.mean(ln_mean)
            ln_std = np.mean(ln_std)

            # manual logging
            eval_dict = {
                "OP/eval_rew_mean": rew_mean,
                "OP/eval_rew_std": rew_std,
                "OP/eval_ln_mean": ln_mean,
                "OP/eval_ln_std": ln_std,
            }
            self.evaluator.write_log(
                eval_dict, iter_idx=int(e * self._step_per_epoch + self._step_per_epoch)
            )

            self.last_reward_mean.append(rew_mean)
            self.last_reward_std.append(rew_std)

            self.save_model(e + 1)
            torch.cuda.empty_cache()

        self.policy.eval()
        self.logger.print(
            "total OP training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

        return first_final_epoch

    def evaluate(self, epoch: int = 5):
        self.evaluator.log_interval = 1
        first_init_epoch = self._init_epoch
        first_final_epoch = self._init_epoch + epoch
        ### Eval Loop
        for e in trange(first_init_epoch, first_final_epoch, desc=f"OP Eval Epoch"):
            self.policy.eval()
            rew_mean = np.zeros((self.policy._num_options,))
            rew_std = np.zeros((self.policy._num_options,))
            ln_mean = np.zeros((self.policy._num_options,))
            ln_std = np.zeros((self.policy._num_options,))
            for z in trange(self.policy._num_options, desc=f"Evaluation", leave=False):
                eval_dict = self.evaluator(
                    self.policy,
                    epoch=e + 1,
                    iter_idx=int(e * self._step_per_epoch + self._step_per_epoch),
                    idx=z,
                    name1=self._val_options[z],
                    dir_name="OP",
                    write_log=False,  # since OP needs to write log of average of all options
                    grid_type=self.grid_type,
                )
                rew_mean[z] = eval_dict["rew_mean"]
                rew_std[z] = eval_dict["rew_std"]
                ln_mean[z] = eval_dict["ln_mean"]
                ln_std[z] = eval_dict["ln_std"]

            rew_mean = np.mean(rew_mean)
            rew_std = np.mean(rew_std)
            ln_mean = np.mean(ln_mean)
            ln_std = np.mean(ln_std)

            # manual logging
            eval_dict = {
                "OP/eval_rew_mean": rew_mean,
                "OP/eval_rew_std": rew_std,
                "OP/eval_ln_mean": ln_mean,
                "OP/eval_ln_std": ln_std,
            }
            self.evaluator.write_log(
                eval_dict, iter_idx=int(e * self._step_per_epoch + self._step_per_epoch)
            )

            torch.cuda.empty_cache()

        return epoch

    def save_model(self, e):
        # save checkpoint
        if e % self.log_interval == 0:
            self.policy.save_model(self.logger.checkpoint_dirs[2], e)

        # save the best model
        if (
            np.mean(self.last_reward_mean) > self.last_max_reward
            and np.mean(self.last_reward_std) <= self.std_limit
        ):
            self.policy.save_model(self.logger.log_dirs[2], e, is_best=True)
            self.last_max_reward = np.mean(self.last_reward_mean)

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values for each key
        sum_dict = {key: 0 for key in dict_list[0].keys()}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                sum_dict[key] += value

        # Calculate the average for each key
        avg_dict = {key: sum_val / len(dict_list) for key, sum_val in sum_dict.items()}

        return avg_dict

    def write_log(self, logging_dict: dict, iter_idx: int):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(int(iter_idx), display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, int(iter_idx))


# model-free policy trainer
# covering option trainer
class OPTrainer2:
    def __init__(
        self,
        policy: OP_Controller,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        evaluator: Evaluator,
        val_options: torch.Tensor,
        epoch: int = 1000,
        init_epoch: int = 0,
        step_per_epoch: int = 1000,
        eval_episodes: int = 10,
        prefix: str = "OP",
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        log_interval: int = 2,
        grid_type: int = 0,
    ) -> None:
        self.policy = policy
        self.sampler = sampler
        self.evaluator = evaluator

        self.logger = logger
        self.writer = writer

        # training parameters
        self._epoch = epoch
        self._init_epoch = init_epoch
        self._step_per_epoch = step_per_epoch

        self.prefix = prefix

        self._eval_episodes = eval_episodes
        self._val_options = val_options
        self._scheduling_epoch = int(self._epoch // 10) if self._epoch >= 10 else None
        self.lr_scheduler = lr_scheduler

        # initialize the essential training components
        self.last_max_reward = 0.0
        self.std_limit = 0.5
        self.num_env_steps = 0

        self.log_interval = log_interval
        self.grid_type = grid_type

    def train(self, z) -> Dict[str, float]:
        start_time = time.time()

        self.last_reward_mean = deque(maxlen=3)
        self.last_reward_std = deque(maxlen=3)

        # train loop
        self.policy.eval()  # policy only has to be train_mode in policy_learn, since sampling needs eval_mode as well.

        first_init_epoch = self._init_epoch
        first_final_epoch = self._epoch
        for e in trange(
            first_init_epoch, first_final_epoch, desc=f"OP PPO Epoch", leave=False
        ):
            ### training loop
            for it in trange(self._step_per_epoch, desc=f"Training", leave=False):
                sample_time = 0
                update_time = 0

                # sample batch
                batch, sampleT = self.sampler.collect_samples(
                    self.policy, idx=z, grid_type=self.grid_type
                )
                sample_time += sampleT

                # update params
                loss_dict, avgRewDict, updateT = self.policy.learn(batch, z)
                update_time += updateT

                # Logging further info
                loss_dict[self.prefix + "_sample_time"] = sample_time
                loss_dict[self.prefix + "_update_time"] = update_time

                self.write_log(loss_dict, iter_idx=int(e * self._step_per_epoch + it))

            # Eval Loop
            eval_dict = self.evaluator(
                self.policy,
                epoch=e + 1,
                iter_idx=int(e * self._step_per_epoch + self._step_per_epoch),
                idx=z,
                name1=self.policy._option_vals[z],
                dir_name=self.prefix,
                write_log=False,  # since OP needs to write log of average of all options
                grid_type=self.grid_type,
            )

            summary_dict = {}
            for k, v in eval_dict.items():
                summary_dict[self.prefix + "_" + k] = v

            # manual logging
            self.evaluator.write_log(
                summary_dict,
                iter_idx=int(e * self._step_per_epoch + self._step_per_epoch),
            )

            self.last_reward_mean.append(eval_dict["rew_mean"])
            self.last_reward_std.append(eval_dict["rew_std"])

            self.save_model(e)
            torch.cuda.empty_cache()

        self.logger.print(
            "total OP2 training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

        return first_final_epoch

    def evaluate(self, z: int, epoch: int = 5):
        self.evaluator.log_interval = 1

        rew_mean = np.zeros((epoch,))
        rew_std = np.zeros((epoch,))
        ln_mean = np.zeros((epoch,))
        ln_std = np.zeros((epoch,))

        ### Eval Loop
        j = 0
        first_init_epoch = self._init_epoch
        first_final_epoch = self._init_epoch + epoch
        for e in trange(
            first_init_epoch, first_final_epoch, desc=f"OP Eval Epoch", leave=False
        ):
            self.policy.eval()

            eval_dict = self.evaluator(
                self.policy,
                epoch=e + 1,
                iter_idx=int(e * self._step_per_epoch + self._step_per_epoch),
                idx=z,
                name1=self._val_options[z],
                dir_name="OP",
                write_log=False,  # since OP needs to write log of average of all options
                grid_type=self.grid_type,
            )

            rew_mean[j] = eval_dict["rew_mean"]
            rew_std[j] = eval_dict["rew_std"]
            ln_mean[j] = eval_dict["ln_mean"]
            ln_std[j] = eval_dict["ln_std"]

            j += 1
            torch.cuda.empty_cache()

        rew_mean = np.mean(rew_mean)
        rew_std = np.mean(rew_std)
        ln_mean = np.mean(ln_mean)
        ln_std = np.mean(ln_std)

        # manual logging
        eval_dict = {
            "OP/eval_rew_mean": rew_mean,
            "OP/eval_rew_std": rew_std,
            "OP/eval_ln_mean": ln_mean,
            "OP/eval_ln_std": ln_std,
        }
        self.evaluator.write_log(
            eval_dict, iter_idx=int(e * self._step_per_epoch + self._step_per_epoch)
        )

        return epoch

    def save_model(self, e):
        # save checkpoint
        if e % self.log_interval == 0:
            self.policy.save_model(self.logger.checkpoint_dirs[2], e)

        # save the best model
        if (
            np.mean(self.last_reward_mean) > self.last_max_reward
            and np.mean(self.last_reward_std) <= self.std_limit
        ):
            self.policy.save_model(self.logger.log_dirs[2], e, is_best=True)
            self.last_max_reward = np.mean(self.last_reward_mean)

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values for each key
        sum_dict = {key: 0 for key in dict_list[0].keys()}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                sum_dict[key] += value

        # Calculate the average for each key
        avg_dict = {key: sum_val / len(dict_list) for key, sum_val in sum_dict.items()}

        return avg_dict

    def write_log(self, logging_dict: dict, iter_idx: int):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(int(iter_idx), display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, int(iter_idx))
