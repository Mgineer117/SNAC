import os
import random
import time
import math

import torch
import torch.nn as nn
import torch.multiprocessing as multiprocessing
import numpy as np
import cv2

from datetime import date
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

today = date.today()


def allocate_values(total, value):
    result = []
    remaining = total

    while remaining >= value:
        result.append(value)
        remaining -= value

    if remaining != 0:
        result.append(remaining)

    return result


def calculate_workers_and_rounds(environments, episodes_per_env, num_cores):
    if episodes_per_env <= 2:
        num_worker_per_env = 1
    elif episodes_per_env > 2:
        num_worker_per_env = episodes_per_env // 2

    # Calculate total number of workers
    total_num_workers = num_worker_per_env * len(environments)

    if total_num_workers > num_cores:
        avail_core_per_env = num_cores // num_worker_per_env

        num_worker_per_round = allocate_values(
            total_num_workers, avail_core_per_env * num_worker_per_env
        )
        num_env_per_round = allocate_values(len(environments), avail_core_per_env)
        rounds = len(num_env_per_round)
    else:
        num_worker_per_round = [total_num_workers]
        num_env_per_round = [len(environments)]
        rounds = 1

    episodes_per_worker = int(episodes_per_env * len(environments) / total_num_workers)
    return num_worker_per_round, num_env_per_round, episodes_per_worker, rounds


class Base:
    def __init__():
        pass

    def initialize(self, episode_num):
        # Preprocess for multiprocessing to avoid CPU overscription and deadlock
        num_workers_per_round, num_env_per_round, episodes_per_worker, rounds = (
            calculate_workers_and_rounds(
                self.training_envs, episode_num, self.num_cores
            )
        )

        self.num_workers_per_round = num_workers_per_round
        self.num_env_per_round = num_env_per_round
        self.total_num_worker = sum(self.num_workers_per_round)
        self.episodes_per_worker = episodes_per_worker
        self.thread_batch_size = self.episodes_per_worker * self.episode_len
        self.num_worker_per_env = int(self.total_num_worker / len(self.training_envs))
        self.rounds = rounds

    def get_reset_data(self, batch_size):
        """
        We create a initialization batch to avoid the daedlocking.
        The remainder of zero arrays will be cut in the end.
        """
        data = dict(
            states=np.zeros(((batch_size,) + self.state_dim)),
            next_states=np.zeros(((batch_size,) + self.state_dim)),
            features=np.zeros(((batch_size, self.feature_dim))),
            actions=np.zeros((batch_size, 1)),
            option_actions=np.zeros((batch_size, 1)),
            rewards=np.zeros((batch_size, 1)),
            terminals=np.zeros((batch_size, 1)),
            logprobs=np.zeros((batch_size, 1)),
        )
        return data

    def set_any_seed(self, env_seed, seed):
        """
        This saves current seed info and calls after stochastic action selection.
        -------------------------------------------------------------------------
        This is to introduce the stochacity in each multiprocessor.
        Without this, the samples from each multiprocessor will be same since the seed was fixed
        """

        temp_seed = env_seed + seed

        # Set the temporary seed
        torch.manual_seed(temp_seed)
        np.random.seed(temp_seed)
        random.seed(temp_seed)

    def collect_samples(
        self,
        policy,
        idx: int = None,
        env_seed: int = 0,
        deterministic: bool = False,
        is_covering_option: bool = False,
    ):
        """
        All sampling and saving to the memory is done in numpy.
        return: dict() with elements in numpy
        """
        t_start = time.time()

        policy_device = policy.device
        policy.to_device(torch.device("cpu"))

        sample_fn = (
            self.collect_trajectory4CoveringOption
            if is_covering_option
            else self.collect_trajectory
        )

        queue = multiprocessing.Manager().Queue()
        env_idx = 0
        worker_idx = 0

        for round_number in range(self.rounds):
            processes = []
            if round_number == self.rounds - 1:
                envs = self.training_envs[env_idx:]
            else:
                envs = self.training_envs[
                    env_idx : env_idx + self.num_env_per_round[round_number]
                ]

            for env in envs:
                workers_for_env = self.num_workers_per_round[round_number] // len(envs)
                for i in range(workers_for_env):
                    if worker_idx == self.total_num_worker - 1:
                        # Main thread process
                        memory = sample_fn(
                            worker_idx,
                            None,
                            env,
                            policy,
                            self.thread_batch_size,
                            self.episode_len,
                            self.episodes_per_worker,
                            idx,
                            env_seed,
                            i,
                            deterministic,
                        )
                    else:
                        # Sub-thread process
                        worker_args = (
                            worker_idx,
                            queue,
                            env,
                            policy,
                            self.thread_batch_size,
                            self.episode_len,
                            self.episodes_per_worker,
                            idx,
                            env_seed,
                            i,
                            deterministic,
                        )
                        p = multiprocessing.Process(target=sample_fn, args=worker_args)
                        processes.append(p)
                        p.start()
                    worker_idx += 1
                env_idx += 1
            for p in processes:
                p.join()

        worker_memories = [None] * (worker_idx - 1)
        for _ in range(worker_idx - 1):
            pid, worker_memory = queue.get()
            worker_memories[pid] = worker_memory

        for worker_memory in worker_memories[::-1]:  # concat in order
            for k in memory:
                memory[k] = np.concatenate((memory[k], worker_memory[k]), axis=0)

        policy.to_device(policy_device)
        t_end = time.time()

        return memory, t_end - t_start


class OnlineSampler(Base):
    def __init__(
        self,
        training_envs,
        state_dim: tuple,
        feature_dim: tuple,
        action_dim: int,
        episode_len: int,
        episode_num: int,
        num_cores: int = None,
    ) -> None:
        super(Base, self).__init__()

        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.episode_num = episode_num
        if isinstance(training_envs, List):
            self.training_envs = training_envs
        else:
            self.training_envs = [training_envs]

        # Preprocess for multiprocessing to avoid CPU overscription and deadlock
        self.num_cores = (
            num_cores if num_cores is not None else multiprocessing.cpu_count()
        )  # torch.get_num_threads() returns appropriate num cpu cores while mp give all available
        num_workers_per_round, num_env_per_round, episodes_per_worker, rounds = (
            calculate_workers_and_rounds(
                self.training_envs, self.episode_num, self.num_cores
            )
        )

        self.num_workers_per_round = num_workers_per_round
        self.num_env_per_round = num_env_per_round
        self.total_num_worker = sum(self.num_workers_per_round)
        self.episodes_per_worker = episodes_per_worker
        self.thread_batch_size = self.episodes_per_worker * self.episode_len
        self.num_worker_per_env = int(self.total_num_worker / len(self.training_envs))
        self.rounds = rounds

        print("Sampling Parameters:")
        print("--------------------")
        print(
            f"Cores (usage)/(given)     : {self.num_workers_per_round[0]}/{self.num_cores} out of {multiprocessing.cpu_count()}"
        )
        print(f"# Environments each Round : {self.num_env_per_round}")
        print(f"Total number of Worker    : {self.total_num_worker}")
        print(f"Episodes per Worker       : {self.episodes_per_worker}")
        torch.set_num_threads(
            1
        )  # enforce one thread for each worker to avoid CPU overscription.

    def collect_trajectory(
        self,
        pid,
        queue,
        env,
        policy: nn.Module,
        thread_batch_size: int,
        episode_len: int,
        episode_num: int,
        idx: int = None,
        env_seed: int = 0,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        # estimate the batch size to hava a large batch
        batch_size = thread_batch_size + episode_len
        data = self.get_reset_data(batch_size=batch_size)  # allocate memory

        current_step = 0
        ep_num = 0

        # For each episode, apply different seed for stochasticity
        if seed is None:
            seed = random.randint(0, 1_000_000)

        if queue is not None:
            # Apply different seeds for multiprocessor's action stochacity
            self.set_any_seed(env_seed, seed)

        while current_step < thread_batch_size:
            if ep_num >= episode_num:
                break

            # env initialization
            s, _ = env.reset(seed=env_seed)
            # s = s["image"]

            t = 0
            while t < episode_len:
                # for t in range(episode_len):
                # sample action
                with torch.no_grad():
                    a, metaData = policy(s, idx, deterministic=deterministic)
                    a = a.cpu().numpy().squeeze()
                    feature = metaData["phi"]

                    option_idx = metaData["z"]  # start feeding option_index

                ### Create an Option Loop
                if metaData["is_option"]:
                    ns, rew, term, trunc, infos = env.step(a)
                    t += 1
                    done = term or trunc

                    option_termination = metaData["termination"]

                    op_rew = rew
                    gamma_count = 1

                    while not (done or option_termination):
                        option_s = ns

                        # env stepping
                        option_a, option_metaData = policy(
                            option_s, option_idx, deterministic=deterministic
                        )
                        option_a = option_a.cpu().numpy().squeeze()

                        ns, rew, term, trunc, infos = env.step(option_a)
                        t += 1

                        op_rew += 0.99**gamma_count * rew

                        gamma_count += 1

                        option_termination = option_metaData["termination"]

                        if gamma_count > 10:
                            option_termination = True

                        done = term or trunc

                    rew = op_rew
                    # print(f"Option time step: {gamma_count}")

                ### Conventional Loop
                else:
                    # env stepping
                    ns, rew, term, trunc, infos = env.step(a)
                    t += 1

                    done = term or trunc

                # saving the data
                data["states"][current_step + t, :, :, :] = s
                data["features"][current_step + t, :] = feature
                data["next_states"][current_step + t, :, :, :] = ns
                data["actions"][current_step + t, :] = a
                data["option_actions"][current_step + t, :] = option_idx
                data["rewards"][current_step + t, :] = rew
                data["terminals"][current_step + t, :] = done
                data["logprobs"][current_step + t, :] = (
                    metaData["logprobs"].detach().numpy()
                )

                s = ns

                if done:
                    # clear log
                    ep_num += 1
                    current_step += t + 1
                    break

        memory = dict(
            states=data["states"].astype(np.float32),
            features=data["features"].astype(np.float32),
            actions=data["actions"].astype(np.float32),
            option_actions=data["option_actions"].astype(np.float32),
            next_states=data["next_states"].astype(np.float32),
            rewards=data["rewards"].astype(np.float32),
            terminals=data["terminals"].astype(np.int32),
            logprobs=data["logprobs"].astype(np.float32),
        )
        end_idx = (
            current_step if current_step < thread_batch_size else thread_batch_size
        )
        for k in memory:
            memory[k] = memory[k][:end_idx]

        if queue is not None:
            queue.put([pid, memory])
        else:
            return memory

    def collect_trajectory4CoveringOption(
        self,
        pid,
        queue,
        env,
        policy: nn.Module,
        thread_batch_size: int,
        episode_len: int,
        episode_num: int,
        idx: int = None,
        env_seed: int = 0,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        # estimate the batch size to hava a large batch
        batch_size = thread_batch_size + episode_len
        data = self.get_reset_data(batch_size=batch_size)  # allocate memory

        current_step = 0
        ep_num = 0

        # For each episode, apply different seed for stochasticity
        if seed is None:
            seed = random.randint(0, 1_000_000)

        if queue is not None:
            # Apply different seeds for multiprocessor's action stochacity
            self.set_any_seed(env_seed, seed)

        while current_step < thread_batch_size:
            if ep_num >= episode_num:
                break

            # env initialization
            s, _ = env.reset(seed=env_seed)
            # s = s["image"]

            t = 0
            is_first_iter = True
            while t < episode_len:
                # for t in range(episode_len):
                # sample action
                with torch.no_grad():
                    a, metaData = policy(s, idx, deterministic=deterministic)
                    a = a.cpu().numpy().squeeze()
                    feature = metaData["phi"]

                    option_idx = metaData["z"]  # start feeding option_index

                ### Create an Option Loop
                if is_first_iter:
                    ns, rew, term, trunc, infos = env.step(a)
                    t += 1
                    done = term or trunc

                    option_termination = metaData["termination"]

                    op_rew = rew
                    gamma_count = 1

                    while not (done or option_termination):
                        option_s = ns

                        # env stepping
                        option_a, option_metaData = policy(
                            option_s, option_idx, deterministic=deterministic
                        )
                        option_a = option_a.cpu().numpy().squeeze()

                        ns, rew, term, trunc, infos = env.step(option_a)
                        t += 1

                        op_rew += 0.99**gamma_count * rew

                        gamma_count += 1

                        option_termination = option_metaData["termination"]

                        if gamma_count >= 9:
                            option_termination = True

                        done = term or trunc

                    rew = op_rew
                    is_first_iter = False
                    # print(f"Option time step: {gamma_count}")

                ### Conventional Loop
                else:
                    # env stepping
                    # forcing random walk after option activation
                    a = torch.randint(0, 4, (1,))
                    ns, rew, term, trunc, infos = env.step(a)
                    t += 1

                    done = term or trunc

                # saving the data
                data["states"][current_step + t, :, :, :] = s
                data["features"][current_step + t, :] = feature
                data["next_states"][current_step + t, :, :, :] = ns
                data["actions"][current_step + t, :] = a
                data["option_actions"][current_step + t, :] = option_idx
                data["rewards"][current_step + t, :] = rew
                data["terminals"][current_step + t, :] = done
                data["logprobs"][current_step + t, :] = (
                    metaData["logprobs"].detach().numpy()
                )

                s = ns

                if done:
                    # clear log
                    ep_num += 1
                    current_step += t + 1
                    break

        memory = dict(
            states=data["states"].astype(np.float32),
            features=data["features"].astype(np.float32),
            actions=data["actions"].astype(np.float32),
            option_actions=data["option_actions"].astype(np.float32),
            next_states=data["next_states"].astype(np.float32),
            rewards=data["rewards"].astype(np.float32),
            terminals=data["terminals"].astype(np.int32),
            logprobs=data["logprobs"].astype(np.float32),
        )
        end_idx = (
            current_step if current_step < thread_batch_size else thread_batch_size
        )
        for k in memory:
            memory[k] = memory[k][:end_idx]

        if queue is not None:
            queue.put([pid, memory])
        else:
            return memory
