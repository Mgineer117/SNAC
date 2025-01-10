import random
import numpy as np
import torch

from math import ceil, floor
from typing import Optional, Union, Tuple, Dict, List


class TrajectoryBuffer:
    def __init__(
        self,
        episode_len: int,
        min_num_trj: int,
        max_num_trj: int,
    ) -> None:
        self.episode_len = episode_len
        self.min_num_trj = min_num_trj
        self.max_num_trj = max_num_trj

        # Using lists to store trajectories
        self.trajectories = []
        self.num_trj = lambda: len(self.trajectories)
        self.full = False

    def decompose(self, batch) -> List[Dict[str, np.ndarray]]:
        """
        Method: Decomposes trajectories in batch in other dimension -> (num_trj * batch_size, d) -> (num_trj, batch_size, d)
        ---------------------------------------------------------------------------------------------------------------------------
        Input: (num_trj * batch_size, d)
        Output: (num_trj, batch_size, d)
        """
        (
            states,
            actions,
            next_states,
            agent_pos,
            next_agent_pos,
            rewards,
            terminals,
            logprobs,
            entropys,
        ) = (
            batch["states"],
            batch["actions"],
            batch["next_states"],
            batch["agent_pos"],
            batch["next_agent_pos"],
            batch["rewards"],
            batch["terminals"],
            batch["logprobs"],
            batch["entropys"],
        )

        trajs = []
        prev_i = 0
        for i, terminal in enumerate(terminals.squeeze()):
            if terminal == 1:
                data = {
                    "states": states[prev_i : i + 1],
                    "actions": actions[prev_i : i + 1],
                    "next_states": next_states[prev_i : i + 1],
                    "agent_pos": agent_pos[prev_i : i + 1],
                    "next_agent_pos": next_agent_pos[prev_i : i + 1],
                    "rewards": rewards[prev_i : i + 1],
                    "terminals": terminals[prev_i : i + 1],
                    "logprobs": logprobs[prev_i : i + 1],
                    "entropys": entropys[prev_i : i + 1],
                }
                trajs.append(data)
                prev_i = i + 1
        return trajs

    def wipe(self):
        self.trajectories = []
        self.full = False

    def push(self, batch: dict, post_process: str | None = None) -> None:
        """
        Method: Push the batch into the data buffer. This saves it as a trajectory
        --------------------------------------------------------------------------------------------
        Input: batch --> dict-type with key of states, actions, next_states, rewards, masks
                        // mask = not done in gym context
        Output: None
        """
        # buffer max criteria
        if len(self.trajectories) >= self.max_num_trj:
            self.full = True
            print("\n+++++Buffer is full now+++++")

        if not self.full:
            trajs = self.decompose(batch)

            if post_process == "nonzero_rewards":
                traj_holder = []
                for trj in trajs:
                    nonzero_indices = np.nonzero(trj["rewards"])[0]
                    if np.any(nonzero_indices):
                        traj_holder.append(trj)

                trajs = traj_holder
            elif post_process == "nonzero_rewards_only":
                # Initialize a dictionary to hold the results
                result_dict = {
                    key: [] for key in trajs[0].keys()
                }  # Assuming all trajs have the same keys

                for trj in trajs:
                    # Find indices where rewards are non-zero
                    nonzero_indices = np.nonzero(trj["rewards"])[0]

                    if nonzero_indices != 0:
                        # Append values at the non-zero indices to the result_dict
                        for key in trj.keys():
                            result_dict[key] = trj[key][nonzero_indices]

                # Convert lists in result_dict to numpy arrays for consistency
                result_dict = {
                    key: np.array(value) for key, value in result_dict.items()
                }
                trajs = [result_dict]

            for traj in trajs:
                print(traj['states'].shape)
                if self.num_trj() < self.max_num_trj:
                    self.trajectories.append(traj)
                else:
                    self.trajectories[self.num_trj() % self.max_num_trj] = traj

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of trajectories from the buffer, ensuring the batch matches the given size.
        """
        
        sampled_batch = self.sample_all()

        for key in sampled_batch.keys():
            batch = sampled_batch[key]
            # Shuffle batch along axis 0
            indices = np.random.permutation(batch.shape[0])
            batch = batch[indices]
            sampled_batch[key] = batch[:batch_size]

        return sampled_batch

    def sample_all(self) -> Dict[str, torch.Tensor]:
        # Sample random trajectories
        sampled_indices = range(0, self.num_trj())

        # Collect sampled data and concatenate
        sampled_data = [self.trajectories[idx] for idx in sampled_indices]

        sampled_batch = {
            "states": np.concatenate([traj["states"] for traj in sampled_data], axis=0),
            "actions": np.concatenate(
                [traj["actions"] for traj in sampled_data], axis=0
            ),
            "next_states": np.concatenate(
                [traj["next_states"] for traj in sampled_data], axis=0
            ),
            "agent_pos": np.concatenate(
                [traj["agent_pos"] for traj in sampled_data], axis=0
            ),
            "next_agent_pos": np.concatenate(
                [traj["next_agent_pos"] for traj in sampled_data], axis=0
            ),
            "rewards": np.concatenate(
                [traj["rewards"] for traj in sampled_data], axis=0
            ),
            "terminals": np.concatenate(
                [traj["terminals"] for traj in sampled_data], axis=0
            ),
            "logprobs": np.concatenate(
                [traj["logprobs"] for traj in sampled_data], axis=0
            ),
            "entropys": np.concatenate(
                [traj["entropys"] for traj in sampled_data], axis=0
            ),
        }

        return sampled_batch
