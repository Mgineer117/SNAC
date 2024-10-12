import uuid

from algorithms.SNAC import SNAC
from algorithms.EigenOption import EigenOption
from algorithms.CoveringOption import CoveringOption
from algorithms.PPO import PPO
from algorithms.FeatureTrain import FeatureTrain


from utils import *

import wandb

wandb.require("core")


def check_all_devices(module):
    devices = {param.device for param in module.parameters()}  # Get all unique devices
    return devices


#########################################################
# Parameter definitions
#########################################################
def train(args, unique_id):
    # call logger
    logger, writer = setup_logger(args, unique_id, seed)

    # start the sf training or import it
    ft = FeatureTrain(logger=logger, writer=writer, args=args)
    sf_network, prev_epoch = ft.train()

    if args.algo_name == "SNAC":
        alg = SNAC(
            sf_network=sf_network,
            prev_epoch=prev_epoch,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "EigenOption":
        alg = EigenOption(
            sf_network=sf_network,
            prev_epoch=prev_epoch,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "CoveringOption":
        alg = CoveringOption(
            sf_network=sf_network,
            prev_epoch=prev_epoch,
            logger=logger,
            writer=writer,
            args=args,
        )
    elif args.algo_name == "PPO":
        alg = PPO(
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

if __name__ == "__main__":
    args = get_args(verbose=False)
    args.seeds = [1, 2, 3]
    seeds = args.seeds
    unique_id = str(uuid.uuid4())[:4]
    for seed in seeds:
        args = get_args()
        args.algo_name: str = "SNAC"
        args.env_name = "CtF1v2"

        # args.SF_epoch = 1
        # args.OP_epoch = 1
        # args.step_per_epoch = 1
        # args.HC_epoch = 1

        # args.import_sf_model = True
        # args.import_op_model = True

        # args.draw_map: bool = True
        seed_all(seed)
        train(args, unique_id)
