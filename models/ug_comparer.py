import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as multiprocessing

from typing import Optional, Dict, List
from tqdm.auto import trange
from collections import deque
from log.wandb_logger import WandbLogger

from utils.sampler import OnlineSampler
from utils.buffer import TrajectoryBuffer

from models.evaulators.sf_evaluator import Evaluator
from models.policy.randomWalk import RandomWalk
from models.policy.optionPolicy import OP_Controller


# Custom scheduler logic for different parameter groups
def custom_lr_scheduler(optimizer, epoch, scheduling_epoch=1):
    if epoch % scheduling_epoch == 0:
        optimizer["phi_optim"].param_groups[0][
            "lr"
        ] *= 0.7  # Reduce learning rate for phi
        # optimizer.param_groups[1]["lr"] *= 0.99  # Reduce learning rate for psi
        # optimizer.param_groups[2]["lr"] *= 0.8  # Reduce learning rate for option
    # pass


def check_all_devices(module):
    devices = {param.device for param in module.parameters()}  # Get all unique devices
    return devices


# model-free policy trainer
class UGComparer:
    """
    Uniform vs Greedy comparison
    """

    def __init__(
        self,
        uniform_policy: RandomWalk,
        option_policy: OP_Controller,
        sampler: OnlineSampler,
        buffer: TrajectoryBuffer,
        logger: WandbLogger,
        writer: SummaryWriter,
        evaluator: Evaluator,
        ### Parmaterers ###
        epoch: int = 1000,
        init_epoch: int = 0,
        step_per_epoch: int = 1000,
        eval_episodes: int = 10,
        log_interval: int = 2,
        env_seed: int = 0,
    ) -> None:
        self.uniform_policy = uniform_policy
        self.option_policy = option_policy

        self.sampler = sampler
        self.buffer = buffer
        self.evaluator = evaluator

        self.logger = logger
        self.writer = writer

        # training parameters
        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._init_epoch = init_epoch
        self._eval_episodes = eval_episodes

        self._num_options = option_policy._num_options

        # initialize the essential training components
        self.last_max_reward = 0.0
        self.last_min_std = 0.0
        self.num_env_steps = 0

        self.log_interval = log_interval
        self.env_seed = env_seed

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        # train loop
        self.uniform_policy.eval()  # policy only has to be train_mode in policy_learn, since sampling needs eval_mode as well.
        self.option_policy.eval()  # policy only has to be train_mode in policy_learn, since sampling needs eval_mode as well.
        self.warm_buffer()

        for z in trange(self._num_options, desc=f"UG Epoch"):
            self.evaluator(
                [self.uniform_policy, self.option_policy],
                epoch=self._init_epoch + z,
                iter_idx=int((self._init_epoch + z) * self._step_per_epoch),
                idx=z,
                dir_name="UG",
                env_seed=self.env_seed,
            )

            kl_div, simple_loss = self.compare(z)
            eval_dict = {
                "UG/kl_div": kl_div.item(),
                "UG/simple_loss": simple_loss.item(),
            }

            self.write_log(
                eval_dict, iter_idx=int((self._init_epoch + z) * self._step_per_epoch)
            )

        self.buffer.wipe()
        self.logger.print(
            "total UG testing time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

    def simple_argmax_loss(self, pred, target):
        # Get the indices of the max values (argmax) along the last dimension
        pred_argmax = torch.argmax(pred, dim=-1)
        target_argmax = torch.argmax(target, dim=-1)

        # Compare the argmax indices and return 0 if they are the same, otherwise 1
        loss = (pred_argmax != target_argmax).float()  # 1 if different, 0 if same
        return loss.mean()  # Average loss over batch if needed

    def compare(self, z):
        batch = self.buffer.sample_all()
        states = torch.from_numpy(batch["states"]).to(torch.float32)
        _, metaData = self.uniform_policy(states, z, deterministic=True)
        uniform_q = metaData["q"]

        _, metaData = self.option_policy(states, z, deterministic=True)
        greedy_q = metaData["q"]

        uniform_prob = F.softmax(uniform_q, dim=-1)  # pred
        greedy_prob = F.softmax(greedy_q, dim=-1)  # target

        kl_div = F.kl_div(uniform_prob.log(), greedy_prob, reduction="batchmean")
        simple_loss = self.simple_argmax_loss(uniform_prob, greedy_prob)

        return kl_div, simple_loss

    def warm_buffer(self):
        # make sure there is nothing there
        self.buffer.wipe()

        # collect enough batch
        print(
            f"\nWarming buffer {self.buffer.num_trj}/{self.buffer.min_num_trj}",
            end="",
        )
        while self.buffer.num_trj < self.buffer.min_num_trj:
            batch, sample_time = self.sampler.collect_samples(
                self.uniform_policy, idx="random", env_seed=self.env_seed
            )
            self.buffer.push(batch)
            print(
                f"\nWarming buffer {self.buffer.num_trj}/{self.buffer.min_num_trj} | sample_time = {sample_time:.2f}s",
                end="",
            )
        print()

    def write_log(self, logging_dict: dict, iter_idx: int):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(int(iter_idx), display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, int(iter_idx))
