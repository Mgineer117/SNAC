import uuid
from algorithms.SNAC import SNAC
from algorithms.EigenOption import EigenOption
from algorithms.CoveringOption import CoveringOption
from algorithms.PPO import PPO
from algorithms.FeatureTrain import FeatureTrain

from utils.call_env import call_env
from utils.utils import save_dim_to_args, setup_logger, seed_all, load_hyperparams
from utils.get_args import get_args

import wandb

wandb.require("core")


#########################################################
# Parameter definitions
#########################################################
def train(args, seed, unique_id):
    """Initiate the training process upon given args

    Args:
        args (arguments): includes all hyperparameters
            - Algorithms: SNAC, EigenOption, CoveringOption, PPO
                - The '+' sign after the algorithm denotes clustering
                    - +: clustering in eigenspace
                    - ++: clustering in value space
        unique_id (int): This is an unique running id for the experiment
    """
    # # call logger
    env = call_env(args)
    save_dim_to_args(env, args)  # given env, save its state and action dim
    logger, writer = setup_logger(args, unique_id, seed)

    if args.algo_name in ("SNAC", "SNAC+", "SNAC++"):
        # start the sf training or import it
        ft = FeatureTrain(env=env, logger=logger, writer=writer, args=args)
        sf_network, prev_epoch = ft.train()
        alg = SNAC(
            env=env,
            sf_network=sf_network,
            prev_epoch=prev_epoch,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name in ("EigenOption", "EigenOption+", "EigenOption++"):
        # start the sf training or import it
        ft = FeatureTrain(env=env, logger=logger, writer=writer, args=args)
        sf_network, prev_epoch = ft.train()
        alg = EigenOption(
            env=env,
            sf_network=sf_network,
            prev_epoch=prev_epoch,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "CoveringOption":
        # start the sf training or import it
        ft = FeatureTrain(env=env, logger=logger, writer=writer, args=args)
        sf_network, prev_epoch = ft.train()
        alg = CoveringOption(
            env=env,
            sf_network=sf_network,
            prev_epoch=prev_epoch,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "PPO":
        alg = PPO(
            env=env,
            logger=logger,
            writer=writer,
            args=args,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algo_name}")

    alg.run()

    wandb.finish()
    writer.close()


#########################################################
# ENV LOOP
#########################################################


def override_args():
    args = get_args(verbose=False)
    file_path = "assets/env_params.json"
    current_params = load_hyperparams(file_path=file_path, env_name=args.env_name)

    # use pre-defined params if no pram given as args
    for k, v in current_params.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    return args


if __name__ == "__main__":
    temp_args = get_args(verbose=False)
    unique_id = str(uuid.uuid4())[:4]

    for seed in temp_args.seeds:
        args = override_args()
        seed_all(seed)
        train(args, seed, unique_id)
