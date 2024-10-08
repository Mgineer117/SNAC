import cv2
import os
import random
import torch
import gymnasium as gym
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.plotter import Plotter
from log.wandb_logger import WandbLogger
from models.evaulators.base_evaluator import Evaluator
import torch.multiprocessing as multiprocessing
from torch.utils.tensorboard import SummaryWriter
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


class PPO_Evaluator(Evaluator):
    def __init__(
        self,
        logger: WandbLogger,
        writer: SummaryWriter,
        training_env: gym.Env,
        plotter: Plotter,
        testing_env=None,
        dir: str = None,
        eigenPlot: bool = False,
        gridPlot: bool = False,
        renderPlot: bool = True,
        eval_ep_num: int = 1,
        log_interval: int = 1,
    ):
        super().__init__(
            logger=logger,
            writer=writer,
            training_env=training_env,
            testing_env=testing_env,
            eval_ep_num=eval_ep_num,
            log_interval=log_interval,
        )
        self.plotter = plotter

        if dir is not None:
            if eigenPlot:
                self.eigenPlot = True
                self.eigenDir = os.path.join(dir, "eigen")
                os.mkdir(self.eigenDir)
            else:
                self.eigenPlot = False
            if gridPlot:
                self.gridPlot = True
                self.gridDir = os.path.join(dir, "grid")
                os.mkdir(self.gridDir)
            else:
                self.gridPlot = False
            if renderPlot:
                self.renderPlot = True
                self.renderDir = os.path.join(dir, "render")
                os.mkdir(self.renderDir)
                self.recorded_frames = []
            else:
                self.renderPlot = False
        else:
            self.eigenPlot = False
            self.gridPlot = False
            self.renderPlot = False

    def eval_loop(
        self,
        env,
        policy: nn.Module,
        epoch: int,
        idx: int = None,
        name1: str = None,
        name2: str = None,
        name3: str = None,
        env_seed: int = 0,
        seed: int = None,
        queue=None,
    ) -> Dict[str, List[float]]:
        ep_buffer = []
        if queue is not None:
            self.set_any_seed(env_seed, seed)

        for num_episodes in range(self.eval_ep_num):
            self.update_render_criteria(epoch, num_episodes)

            # logging initialization
            ep_reward, ep_length = 0, 0

            # env initialization
            s, _ = env.reset(seed=env_seed)

            if self.eigenCriteria:
                self.init_grid(env)

            done = False
            while not done:
                with torch.no_grad():
                    a, phi_dict = policy(s, idx, deterministic=True)
                    a = a.cpu().numpy().squeeze()

                ns, rew, term, trunc, _ = env.step(a)
                done = term or trunc

                s = ns
                ep_reward += rew
                ep_length += 1

                # Update the grid
                if self.eigenCriteria:
                    if hasattr(env.env, "agent_pos"):
                        agent_pos = env.get_wrapper_attr("agent_pos")
                    elif hasattr(env.env, "agents"):
                        agent_pos = env.get_wrapper_attr("agents")[0].pos
                    else:
                        raise ValueError("No agent position information.")
                    self.update_grid(
                        agent_pos,
                        phi_dict["phi_r"],
                        phi_dict["phi_s"],
                        phi_dict["q"],
                    )
                # Update the render
                if self.renderCriteria:
                    img = env.render()
                    self.recorded_frames.append(img)

                if done:
                    if self.eigenCriteria:
                        self.plotter.plotEigenFunction1(
                            eigenvectors=policy._options[:, idx]
                            .clone()
                            .detach()
                            .numpy()
                            .T,
                            dir=self.eigenDir,
                            epoch=epoch,
                        )

                    if self.gridCriteria:
                        self.plotter.plotFeature(
                            self.grid,
                            self.grid_r,
                            self.grid_s,
                            self.grid_v,
                            self.grid_q,
                            dir=self.gridDir,
                            epoch=epoch,
                        )

                    if self.renderCriteria:
                        width = self.recorded_frames[0].shape[0]
                        height = self.recorded_frames[0].shape[1]
                        self.plotter.plotRendering(
                            self.recorded_frames,
                            dir=self.renderDir,
                            epoch=str(epoch),
                            width=width,
                            height=height,
                        )
                        self.recorded_frames = []

                    ep_buffer.append({"reward": ep_reward, "ep_length": ep_length})

        reward_list = [ep_info["reward"] for ep_info in ep_buffer]
        length_list = [ep_info["ep_length"] for ep_info in ep_buffer]

        rew_mean, rew_std = np.mean(reward_list), np.std(reward_list)
        ln_mean, ln_std = np.mean(length_list), np.std(length_list)

        if queue is not None:
            queue.put([rew_mean, rew_std, ln_mean, ln_std])
        else:
            return rew_mean, rew_std, ln_mean, ln_std

    def update_render_criteria(self, epoch, num_episodes):
        basisCriteria = epoch % self.log_interval == 0 and num_episodes == 0
        self.eigenCriteria = basisCriteria and self.eigenPlot
        self.gridCriteria = basisCriteria and self.gridPlot
        self.renderCriteria = basisCriteria and self.renderPlot

    def init_grid(self, env):
        self.grid = np.copy(env.render()).astype(np.float32) / 255.0

        self.grid_r = np.zeros(self.grid.shape)
        self.grid_s = np.zeros(self.grid.shape)
        self.grid_v = np.zeros(self.grid.shape)
        self.grid_q = np.zeros(self.grid.shape)

    def update_grid(self, coord, phi_r, phi_s, q):
        ### create image
        phi_r, phi_s = phi_r.cpu().numpy(), phi_s.cpu().numpy()

        phi_r = np.sum(phi_r) / phi_r.shape[-1]
        phi_s = np.sum(phi_s) / phi_s.shape[-1]

        colormap = cm.get_cmap("gray")  # Choose a colormap
        color_r = colormap(phi_r * 5)[:3]  # Get RGB values
        color_s = colormap(phi_s * 2)[:3]  # Get RGB values
        color_q = colormap(q)[:3]  # Get RGB values
        # coord =
        coordx = [
            coord[0] * self.tile_size + 1,
            coord[0] * self.tile_size + self.tile_size - 1,
        ]
        coordy = [
            coord[1] * self.tile_size + 1,
            coord[1] * self.tile_size + self.tile_size - 1,
        ]

        self.grid_r[coordx[0] : coordx[1], coordy[0] : coordy[1], :] = color_r
        self.grid_s[coordx[0] : coordx[1], coordy[0] : coordy[1], :] = color_s
        self.grid_v[coordx[0] : coordx[1], coordy[0] : coordy[1], :] += (
            0.01,
            0.01,
            0.01,
        )
        self.grid_q[coordx[0] : coordx[1], coordy[0] : coordy[1], :] = color_q

    def plot_options(self, S, V):
        self.plotter.plotEigenFunctionAll(S.numpy(), V.numpy())
