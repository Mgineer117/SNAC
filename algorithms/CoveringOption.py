import torch
import torch.nn as nn
import random
import uuid
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gym_multigrid.envs.fourrooms import FourRooms

from models.evaulators import (
    SF_Evaluator,
    OP_Evaluator2,
    UG_Evaluator,
    HC_Evaluator,
)
from models import SFTrainer, OPTrainer2, HCTrainer
from utils import *


class CoveringOption:
    """
    The difference from SNAC is two-fold:
        - it does not have reward-predictive feature
        - it does only pick top n number of eigenvectors
            - this heuristics ignores other info of eigs
    """

    def __init__(self, env: gym.Env, buffer, sampler, logger, writer, args):
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
        self.op_evaluator = OP_Evaluator2(
            logger=logger,
            writer=writer,
            training_env=env,
            plotter=self.plotter,
            dir=self.op_path,
            log_interval=args.log_interval,
            eval_ep_num=5,
        )
        self.ug_evaluator = UG_Evaluator(
            logger=logger,
            writer=writer,
            training_env=env,
            plotter=self.plotter,
            dir=self.ug_path,
            log_interval=args.log_interval,
        )
        self.hc_evaluator = HC_Evaluator(
            logger=logger,
            writer=writer,
            training_env=env,
            plotter=self.plotter,
            dir=self.hc_path,
            log_interval=args.log_interval,
            eval_ep_num=5,
        )

    def run(self):
        """
        Consecutively run all pipelines
        """
        self.train_sf()
        self.train_op()
        self.train_hc()

    def train_sf(self):
        """
        This trains the SF netowk. This includes training of CNN as feature extractor
        and Psi_rw network as a supervised approach as a evaluation metroc for later use.
        ----------------------------------------------------------------------------------
        Input:
            - None
        Return:
            - None
        """
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

    def train_op(self):
        """
        This is more sophisticated approach since it has to iteratively update SF matrix
        ----------------------------------------------
        High-level idea:
            - Find the first eigenvectors with pure random walk
            for i in range(n)
                - With that eigenvector, collect samples and concat to the original SF matrix
                    - Decompose cat SF matrix to discover better naviagtional vector
                - Train OptionPolicy for the vectors
        ----------------------------------------------
        """
        self.option_vals = torch.zeros((self.args.num_vector))
        self.options = torch.zeros((self.args.num_vector, self.args.sf_dim))

        # train two vec only
        self.op_network = call_opNetwork(
            self.sf_network, self.option_vals, self.options, self.args
        )
        print_model_summary(self.op_network, model_name="OP model")

        if not self.args.import_op_model:
            app_trj_num = int(100 / self.args.num_vector)

            ### get first vector with random walk
            batch = self.collect_batch(self.sf_network, app_trj_num=100, idx=None)
            with torch.no_grad():
                S, V = self.get_vector(batch)

            # update option in op network
            self.option_vals[0:2] = S
            self.options[0:2, :] = V
            self.op_network._option_vals = self.option_vals
            self.op_network._options = nn.Parameter(
                self.options.to(torch.float32).to(self.args.device)
            )

            self.train_op_network(vec_idx=0)
            batch = None
            for idx in range(1, int(self.args.num_vector / 2)):
                vec_idx = idx * 2

                new_batch1 = self.collect_batch(
                    self.op_network, app_trj_num=app_trj_num, idx=vec_idx
                )
                new_batch2 = self.collect_batch(
                    self.op_network, app_trj_num=app_trj_num, idx=vec_idx + 1
                )
                batch = self.cat_batch(batch, new_batch1, new_batch2)
                with torch.no_grad():
                    S, V = self.get_vector(batch)
                self.option_vals[vec_idx : vec_idx + 2] = S
                self.options[vec_idx : vec_idx + 2, :] = V
                self.op_network._option_vals = self.option_vals
                self.op_network._options = nn.Parameter(
                    self.options.to(torch.float32).to(self.args.device)
                )
                self.train_op_network(vec_idx=vec_idx)
        else:
            final_epoch = self.curr_epoch + self.args.num_vector * self.args.OP_epoch
            self.curr_epoch += final_epoch

    def train_hc(self):
        """
        Train Hierarchical Controller to compute optimal policy that alternates between
        options and the random walk.
        """
        self.hc_network = call_hcNetwork(
            self.sf_network.feaNet, self.op_network, self.args
        )
        print_model_summary(self.hc_network, model_name="HC model")
        if not self.args.import_hc_model:
            hc_trainer = HCTrainer(
                policy=self.hc_network,
                sampler=self.sampler,
                logger=self.logger,
                writer=self.writer,
                evaluator=self.hc_evaluator,
                prefix="HC",
                epoch=self.curr_epoch + self.args.HC_epoch,
                init_epoch=self.curr_epoch,
                step_per_epoch=self.args.step_per_epoch,
                eval_episodes=self.args.eval_episodes,
                log_interval=self.args.log_interval,
                env_seed=self.args.env_seed,
            )
            hc_trainer.train()
        self.curr_epoch += self.args.HC_epoch

    def collect_batch(
        self, policy: nn.Module, app_trj_num: int = 10, idx: int | None = None
    ):
        """
        Colelct batch with the given policy for the given number of trajectories.
        -------------------------------------------------------------------------
        Input:
            - policy: decision maker (sf: random walk, op: option walk)
            - idx: if op_network is given, idx denotes the option index to activate
        Return:
            - batch: batch collected using option (10 time steps) and remaining as Random walk (90 steps)
                - To concat the batch to improve the diffusion SF matrix
        """
        option_buffer = TrajectoryBuffer(
            min_num_trj=app_trj_num, max_num_trj=200, device=self.args.device
        )

        while option_buffer.num_trj < option_buffer.min_num_trj:
            batch, sample_time = self.sampler.collect_samples(
                policy,
                env_seed=self.args.env_seed,
                idx=idx,
                is_covering_option=True,
            )
            option_buffer.push(batch)
        batch = option_buffer.sample_all()
        option_buffer.wipe()

        return batch

    def get_vector(self, batch):
        """
        This discovers vectors from the feature set given batch.
        Additionally, we use both (+/-) of vectors (top 1 vector => 2 vectors)
        """
        features = (
            torch.from_numpy(batch["features"]).to(torch.float32).to(self.args.device)
        )
        terminals = (
            torch.from_numpy(batch["terminals"]).to(torch.float32).to(self.args.device)
        )

        with torch.no_grad():
            psi = estimate_psi(
                features, terminals, self.args.gamma, device=self.args.device
            )

            _, S, V = torch.svd(psi)  # S: max - min

        S = S[0].unsqueeze(0)
        V = V[0, :].unsqueeze(0)

        S = torch.cat((S, -S), axis=0)
        V = torch.cat((V, -V), axis=0)

        return S, V

    def cat_batch(self, batch, new_batch1, new_batch2):
        """
        This is to concat the previously collected SF diffusive matrix with newly collected batch
        to improve its diffusion of the given domain. This is a particular approach of CoveringOption
        that is usually adopted for hardly-exploratory environments.
        """
        if batch is None:
            batch = {}
            batch["features"] = np.concatenate(
                (new_batch1["features"], new_batch2["features"]),
                axis=0,
            )
            batch["terminals"] = np.concatenate(
                (new_batch1["terminals"], new_batch2["terminals"]),
                axis=0,
            )
        else:
            batch["features"] = np.concatenate(
                (batch["features"], new_batch1["features"], new_batch2["features"]),
                axis=0,
            )
            batch["terminals"] = np.concatenate(
                (batch["terminals"], new_batch1["terminals"], new_batch2["terminals"]),
                axis=0,
            )
        return batch

    def train_op_network(self, vec_idx):
        """
        This trains the OP network for the given indices. Initially, the OP network is set with zero eigenvectors,
        but as we discover and update new vectors, we continuously refine them.
        The OP network is trained to identify a more effective diffusion matrix over time
        """
        for z in range(vec_idx, vec_idx + 2):
            op_trainer = OPTrainer2(
                policy=self.op_network,
                sampler=self.sampler,
                logger=self.logger,
                writer=self.writer,
                evaluator=self.op_evaluator,
                epoch=self.curr_epoch + self.args.OP_epoch,
                init_epoch=self.curr_epoch,
                psi_epoch=self.args.Psi_epoch,
                step_per_epoch=self.args.step_per_epoch,
                eval_episodes=self.args.eval_episodes,
                prefix="OP/" + str(z),
                log_interval=self.args.log_interval,
                env_seed=self.args.env_seed,
            )

            op_trainer.train(z)
            self.curr_epoch += self.args.OP_epoch
