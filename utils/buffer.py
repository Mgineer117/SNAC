import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict, List


class TrajectoryBuffer:
    def __init__(
        self, min_num_trj: int, max_num_trj: int, device: str = torch.device("cpu")
    ) -> None:
        self.min_num_trj = min_num_trj
        self.max_num_trj = max_num_trj
        self.device = device

        # Using lists to store trajectories
        self.trajectories = []
        self.num_trj = 0

    def decompose(self, batch) -> List[Dict[str, np.ndarray]]:
        """
        Method: Decomposes trajectories in batch in other dimension -> (num_trj * batch_size, d) -> (num_trj, batch_size, d)
        ---------------------------------------------------------------------------------------------------------------------------
        Input: (num_trj * batch_size, d)
        Output: (num_trj, batch_size, d)
        """
        (
            states,
            features,
            actions,
            next_states,
            rewards,
            terminals,
            logprobs,
        ) = (
            batch["states"],
            batch["features"],
            batch["actions"],
            batch["next_states"],
            batch["rewards"],
            batch["terminals"],
            batch["logprobs"],
        )

        trajs = []
        prev_i = 0
        for i, terminal in enumerate(terminals.squeeze()):
            if terminal == 1:
                data = {
                    "states": states[prev_i : i + 1],
                    "features": features[prev_i : i + 1],
                    "actions": actions[prev_i : i + 1],
                    "next_states": next_states[prev_i : i + 1],
                    "rewards": rewards[prev_i : i + 1],
                    "terminals": terminals[prev_i : i + 1],
                    "logprobs": logprobs[prev_i : i + 1],
                }
                trajs.append(data)
                prev_i = i + 1
        return trajs

    def wipe(self):
        self.trajectories = []
        self.num_trj = 0

    def push(self, batch: dict) -> None:
        """
        Method: Push the batch into the data buffer. This saves it as a trajectory
        --------------------------------------------------------------------------------------------
        Input: batch --> dict-type with key of states, actions, next_states, rewards, masks
                        // mask = not done in gym context
        Output: None
        """
        trajs = self.decompose(batch)

        for traj in trajs:
            if self.num_trj < self.max_num_trj:
                self.trajectories.append(traj)
            else:
                self.trajectories[self.num_trj % self.max_num_trj] = traj
            self.num_trj += 1

    def sample(self, num_traj: int) -> Dict[str, torch.Tensor]:
        if num_traj > self.num_trj:
            num_traj = self.num_trj

        # Sample random trajectories
        sampled_indices = np.random.choice(
            min(self.num_trj, self.max_num_trj), num_traj, replace=False
        )

        # Collect sampled data and concatenate
        sampled_data = [self.trajectories[idx] for idx in sampled_indices]

        sampled_batch = {
            "states": np.concatenate([traj["states"] for traj in sampled_data], axis=0),
            "features": np.concatenate(
                [traj["features"] for traj in sampled_data], axis=0
            ),
            "actions": np.concatenate(
                [traj["actions"] for traj in sampled_data], axis=0
            ),
            "next_states": np.concatenate(
                [traj["next_states"] for traj in sampled_data], axis=0
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
        }

        return sampled_batch

    def sample_all(self) -> Dict[str, torch.Tensor]:
        # Sample random trajectories
        sampled_indices = range(0, self.num_trj)

        # Collect sampled data and concatenate
        sampled_data = [self.trajectories[idx] for idx in sampled_indices]

        sampled_batch = {
            "states": np.concatenate([traj["states"] for traj in sampled_data], axis=0),
            "features": np.concatenate(
                [traj["features"] for traj in sampled_data], axis=0
            ),
            "actions": np.concatenate(
                [traj["actions"] for traj in sampled_data], axis=0
            ),
            "next_states": np.concatenate(
                [traj["next_states"] for traj in sampled_data], axis=0
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
        }

        return sampled_batch
