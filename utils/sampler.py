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
            agent_pos=np.zeros(((batch_size, 2))),
            next_agent_pos=np.zeros(((batch_size, 2))),
            rewards=np.zeros((batch_size, 1)),
            terminals=np.zeros((batch_size, 1)),
            logprobs=np.zeros((batch_size, 1)),
        )
        return data

    def set_any_seed(self, seed, pid):
        """
        This saves current seed info and calls after stochastic action selection.
        -------------------------------------------------------------------------
        This is to introduce the stochacity in each multiprocessor.
        Without this, the samples from each multiprocessor will be same since the seed was fixed
        """

        temp_seed = seed + pid

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
        is_option: bool = False,
        is_covering_option: bool = False,
    ):
        """
        All sampling and saving to the memory is done in numpy.
        return: dict() with elements in numpy
        """
        t_start = time.time()

        policy_device = policy.device
        policy.to_device(torch.device("cpu"))

        # select appropriate sampler function
        if is_covering_option and not is_option:
            sample_fn = self.collect_trajectory4CoveringOption
        elif is_option and not is_covering_option:
            sample_fn = self.collect_trajectory4Option
        else:
            sample_fn = self.collect_trajectory

        queue = multiprocessing.Manager().Queue()
        env_idx = 0
        worker_idx = 0

        # iterate over rounds
        for round_number in range(self.rounds):
            processes = []
            if round_number == self.rounds - 1:
                envs = self.training_envs[env_idx:]
            else:
                envs = self.training_envs[
                    env_idx : env_idx + self.num_env_per_round[round_number]
                ]

            # iterate over envs
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
        min_option_length: int,
        min_cover_option_length: int,
        episode_len: int,
        episode_num: int,
        num_cores: int = None,
    ) -> None:
        super(Base, self).__init__()
        """
        !!! This can even handle multi-envoronments !!!
        This computes the ""very"" appropriate parameter for the Monte-Carlo sampling
        given the number of episodes and the given number of cores the runner specified.
        ---------------------------------------------------------------------------------
        Rounds: This gives several rounds when the given sampling load exceeds the number of threads
        the task is assigned. 
        This assigned appropriate parameters assuming one worker work with 2 trajectories.
        """

        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.min_option_length = min_option_length
        self.min_cover_option_length = min_cover_option_length
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

        print("====================")
        print("Sampling Parameters:")
        print("====================")
        print(
            f"Cores (usage)/(given)     : {self.num_workers_per_round[0]}/{self.num_cores} out of {multiprocessing.cpu_count()}"
        )
        print(f"# Environments each Round : {self.num_env_per_round}")
        print(f"Total number of Worker    : {self.total_num_worker}")
        print(f"Episodes per Worker       : {self.episodes_per_worker}")

        # enforce one thread for each worker to avoid CPU overscription.
        torch.set_num_threads(1)

    def collect_trajectory(
        self,
        pid,
        queue,
        env,
        policy: nn.Module,
        thread_batch_size: int,
        episode_len: int,
        episode_num: int,
        idx: int | None = None,
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
            self.set_any_seed(seed, pid)

        while current_step < thread_batch_size:
            if ep_num >= episode_num:
                break

            # env initialization
            obs, _ = env.reset(seed=env_seed)
            s = obs["observation"]
            agent_pos = obs["agent_pos"]

            t = 0
            while t < episode_len:
                # for t in range(episode_len):
                with torch.no_grad():
                    a, metaData = policy(obs, idx, deterministic=deterministic)
                    a = a.cpu().numpy().squeeze()
                    feature = metaData["phi"]

                    # env stepping
                    next_obs, rew, term, trunc, infos = env.step(a)
                    ns = next_obs["observation"]
                    next_agent_pos = next_obs["agent_pos"]

                    done = term or trunc

                # saving the data
                data["states"][current_step + t, :, :, :] = s
                data["features"][current_step + t, :] = feature
                data["next_states"][current_step + t, :, :, :] = ns
                data["actions"][current_step + t, :] = a
                data["option_actions"][current_step + t, :] = None
                data["agent_pos"][current_step + t, :] = agent_pos
                data["next_agent_pos"][current_step + t, :] = next_agent_pos
                data["rewards"][current_step + t, :] = rew
                data["terminals"][current_step + t, :] = done
                data["logprobs"][current_step + t, :] = (
                    metaData["logprobs"].detach().numpy()
                )

                obs = next_obs
                s = ns
                agent_pos = next_agent_pos

                t += 1

                if done:
                    # clear log
                    ep_num += 1
                    current_step += t + 1
                    t = 0
                    break

        memory = dict(
            states=data["states"].astype(np.float32),
            features=data["features"].astype(np.float32),
            actions=data["actions"].astype(np.float32),
            option_actions=data["option_actions"].astype(np.float32),
            next_states=data["next_states"].astype(np.float32),
            agent_pos=data["agent_pos"].astype(np.float32),
            next_agent_pos=data["next_agent_pos"].astype(np.float32),
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

    def collect_trajectory4Option(
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
            self.set_any_seed(seed, pid)

        while current_step < thread_batch_size:
            if ep_num >= episode_num:
                break

            # env initialization
            obs, _ = env.reset(seed=env_seed)
            s = obs["observation"]
            agent_pos = obs["agent_pos"]

            t = 0
            while t < episode_len:
                # for t in range(episode_len):
                # sample action
                with torch.no_grad():
                    a, metaData = policy(obs, idx, deterministic=deterministic)
                    a = a.cpu().numpy().squeeze()
                    feature = metaData["phi"]

                    option_idx = metaData["z"]  # start feeding option_index

                ### Create an Option Loop
                if metaData["is_option"]:
                    next_obs, rew, term, trunc, infos = env.step(a)
                    ns = next_obs["observation"]
                    next_agent_pos = next_obs["agent_pos"]

                    done = term or trunc

                    # option_termination = metaData["termination"]

                    op_rew = rew
                    step_count = 1

                    option_termination = False
                    while not (done or option_termination):
                        option_s = next_obs

                        # env stepping
                        option_a, _ = policy(
                            option_s, option_idx, deterministic=deterministic
                        )
                        option_a = option_a.cpu().numpy().squeeze()

                        next_obs, rew, term, trunc, infos = env.step(option_a)
                        ns = next_obs["observation"]
                        next_agent_pos = next_obs["agent_pos"]

                        op_rew += 0.99**step_count * rew
                        step_count += 1

                        option_termination = (
                            True if step_count >= self.min_option_length else False
                        )
                        done = term or trunc
                    # print(f"step count: {step_count} with option_idx {option_idx}")
                    rew = op_rew

                ### Conventional Loop
                else:
                    step_count = 1  # dummy
                    # env stepping
                    next_obs, rew, term, trunc, infos = env.step(a)
                    ns = next_obs["observation"]
                    next_agent_pos = next_obs["agent_pos"]

                    done = term or trunc

                # saving the data
                data["states"][current_step + t, :, :, :] = s
                data["features"][current_step + t, :] = feature
                data["next_states"][current_step + t, :, :, :] = ns
                data["actions"][current_step + t, :] = a
                data["option_actions"][current_step + t, :] = option_idx
                data["agent_pos"][current_step + t, :] = agent_pos
                data["next_agent_pos"][current_step + t, :] = next_agent_pos
                data["rewards"][current_step + t, :] = rew
                data["terminals"][current_step + t, :] = done
                data["logprobs"][current_step + t, :] = (
                    metaData["logprobs"].detach().numpy()
                )

                obs = next_obs
                s = ns
                agent_pos = next_agent_pos

                t += step_count

                if done:
                    # clear log
                    ep_num += 1
                    current_step += t + 1
                    t = 0
                    break

        memory = dict(
            states=data["states"].astype(np.float32),
            features=data["features"].astype(np.float32),
            actions=data["actions"].astype(np.float32),
            option_actions=data["option_actions"].astype(np.float32),
            next_states=data["next_states"].astype(np.float32),
            agent_pos=data["agent_pos"].astype(np.float32),
            next_agent_pos=data["next_agent_pos"].astype(np.float32),
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
        """
        This is separately written since the CoveringOption requires
        iterative update with batch that first uses eigen-teleport to move
        farthest location and do random walk to capture better diffusion property
        of the given environment. If it does not make sense, we guide the reader to
        run the code a blackbox or please refer to section (5.2) of
        Machado, Marlos C., et al. "Temporal abstraction in reinforcement learning with the successor representation."
        """
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
            self.set_any_seed(seed, pid)

        while current_step < thread_batch_size:
            if ep_num >= episode_num:
                break

            # env initialization
            obs, _ = env.reset(seed=env_seed)
            s = obs["observation"]
            agent_pos = obs["agent_pos"]

            t = 0
            is_first_iter = True
            while t < episode_len:
                # for t in range(episode_len):
                # sample action
                with torch.no_grad():
                    a, metaData = policy(obs, idx, deterministic=deterministic)
                    a = a.cpu().numpy().squeeze()
                    feature = metaData["phi"]

                    option_idx = metaData["z"]  # start feeding option_index

                ### Create an Option Loop
                if is_first_iter:
                    next_obs, rew, term, trunc, infos = env.step(a)
                    ns = next_obs["observation"]
                    next_agent_pos = next_obs["agent_pos"]

                    done = term or trunc

                    # option_termination = metaData["termination"]

                    op_rew = rew
                    step_count = 1

                    option_termination = False
                    while not (done or option_termination):
                        option_s = next_obs

                        # env stepping
                        option_a, _ = policy(
                            option_s, option_idx, deterministic=deterministic
                        )
                        option_a = option_a.cpu().numpy().squeeze()

                        next_obs, rew, term, trunc, infos = env.step(option_a)
                        ns = next_obs["observation"]
                        next_agent_pos = next_obs["agent_pos"]

                        op_rew += 0.99**step_count * rew
                        step_count += 1

                        option_termination = (
                            True
                            if step_count >= self.min_cover_option_length
                            else False
                        )
                        done = term or trunc

                    rew = op_rew
                    is_first_iter = False
                ### Conventional Loop
                else:
                    step_count = 1
                    # env stepping
                    # forcing random walk after option activation
                    a = torch.randint(0, self.action_dim, (1,)).squeeze()

                    next_obs, rew, term, trunc, infos = env.step(a)
                    ns = next_obs["observation"]
                    next_agent_pos = next_obs["agent_pos"]

                    done = term or trunc

                # saving the data
                data["states"][current_step + t, :, :, :] = s
                data["features"][current_step + t, :] = feature
                data["next_states"][current_step + t, :, :, :] = ns
                data["actions"][current_step + t, :] = a
                data["option_actions"][current_step + t, :] = option_idx
                data["agent_pos"][current_step + t, :] = agent_pos
                data["next_agent_pos"][current_step + t, :] = next_agent_pos
                data["rewards"][current_step + t, :] = rew
                data["terminals"][current_step + t, :] = done
                data["logprobs"][current_step + t, :] = (
                    metaData["logprobs"].detach().numpy()
                )

                obs = next_obs
                s = ns
                agent_pos = next_agent_pos

                t += step_count

                if done:
                    # clear log
                    ep_num += 1
                    current_step += t + 1
                    t = 0
                    break

        memory = dict(
            states=data["states"].astype(np.float32),
            features=data["features"].astype(np.float32),
            actions=data["actions"].astype(np.float32),
            option_actions=data["option_actions"].astype(np.float32),
            next_states=data["next_states"].astype(np.float32),
            agent_pos=data["agent_pos"].astype(np.float32),
            next_agent_pos=data["next_agent_pos"].astype(np.float32),
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
