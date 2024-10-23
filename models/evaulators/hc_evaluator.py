import cv2
import os
import random
import torch
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


class HC_Evaluator(Evaluator):
    def __init__(
        self,
        logger: WandbLogger,
        writer: SummaryWriter,
        training_env,
        plotter: Plotter,
        testing_env=None,
        dir: str = None,
        gridPlot: bool = True,
        renderPlot: bool = False,
        min_option_length: int = 3,
        eval_ep_num: int = 1,
        log_interval: int = 1,
    ):
        super(HC_Evaluator, self).__init__(
            logger=logger,
            writer=writer,
            training_env=training_env,
            testing_env=testing_env,
            eval_ep_num=eval_ep_num,
            log_interval=log_interval,
        )
        self.plotter = plotter
        self.min_option_length = min_option_length
        if dir is not None:
            if gridPlot:
                self.gridPlot = True
                self.gridDir = os.path.join(dir, "grid")
                os.mkdir(self.gridDir)
                self.path = []
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

        option_indices_list = []
        for num_episodes in range(self.eval_ep_num):
            self.update_render_criteria(epoch, num_episodes)

            # logging initialization
            ep_reward, ep_length = 0, 0

            # env initialization
            obs, _ = env.reset(seed=env_seed)

            if self.gridCriteria:
                self.init_grid(env)

            option_indices = []
            done = False
            while not done:
                with torch.no_grad():
                    a, metaData = policy(obs, idx, deterministic=True)
                    a = a.cpu().numpy().squeeze()
                    option_idx = metaData["z"]  # start feeding option_index
                    option_indices.append(option_idx)

                ### Create an Option Loop
                if metaData["is_option"]:
                    # Update the grid
                    if self.gridCriteria:
                        self.get_agent_pos(env)

                    next_obs, rew, term, trunc, infos = env.step(a)

                    done = term or trunc

                    op_rew = rew
                    step_count = 1

                    option_termination = False
                    while not (done or option_termination):
                        option_s = next_obs

                        # env stepping
                        option_a, _ = policy(option_s, option_idx, deterministic=True)
                        option_a = option_a.cpu().numpy().squeeze()

                        # Update the grid
                        if self.gridCriteria:
                            self.get_agent_pos(env)

                        next_obs, rew, term, trunc, infos = env.step(option_a)

                        op_rew += 0.99**step_count * rew
                        step_count += 1

                        option_termination = (
                            True if step_count >= self.min_option_length else False
                        )
                        done = term or trunc

                    rew = op_rew

                ### Conventional Loop
                else:
                    # Update the grid
                    if self.gridCriteria:
                        self.get_agent_pos(env)

                    step_count = 1  # dummy
                    # env stepping
                    next_obs, rew, term, trunc, infos = env.step(a)

                    done = term or trunc

                obs = next_obs

                ep_reward += rew
                ep_length += step_count

                # Update the render
                if self.renderCriteria:
                    img = env.render()
                    self.recorded_frames.append(img)

                if done:
                    if self.gridCriteria:
                        # final agent pos
                        self.get_agent_pos(env)

                        self.plotter.plotPath(
                            self.grid,
                            self.path,
                            dir=self.gridDir,
                            epoch=str(epoch),
                        )
                        self.path = []

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

                    option_indices_list.append(option_indices)
                    ep_buffer.append({"reward": ep_reward, "ep_length": ep_length})

        reward_list = [ep_info["reward"] for ep_info in ep_buffer]
        length_list = [ep_info["ep_length"] for ep_info in ep_buffer]

        rew_mean, rew_std = np.mean(reward_list), np.std(reward_list)
        ln_mean, ln_std = np.mean(length_list), np.std(length_list)

        if queue is not None:
            queue.put([rew_mean, rew_std, ln_mean, ln_std])
        else:
            for x in option_indices_list:
                plt.scatter(np.arange(len(x)), x)

            option_figure_path = os.path.join(self.plotter.hc_path, "option_figure")
            if not os.path.exists(option_figure_path):
                os.mkdir(option_figure_path)
            plt.savefig(f"{option_figure_path}/{epoch}_{num_episodes}.png")
            plt.close()
            return rew_mean, rew_std, ln_mean, ln_std

    def update_render_criteria(self, epoch, num_episodes):
        basisCriteria = epoch % self.log_interval == 0 and num_episodes == 0
        self.gridCriteria = basisCriteria and self.gridPlot
        self.renderCriteria = basisCriteria and self.renderPlot

    def init_grid(self, env):
        self.grid = np.copy(env.render()).astype(np.float32) / 255.0

    def get_agent_pos(self, env):
        # Update the grid
        if self.gridCriteria:
            if hasattr(env.env, "agent_pos"):
                self.path.append(env.get_wrapper_attr("agent_pos"))
            elif hasattr(env.env, "agents"):
                self.path.append(env.get_wrapper_attr("agents")[0].pos)
            else:
                raise ValueError("No agent position information.")
