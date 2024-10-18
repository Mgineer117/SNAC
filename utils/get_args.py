"""Define variables and hyperparameters using argparse"""

import argparse
import torch


def list_of_ints(arg):
    """Terminal seed input interpreter"""
    return list(map(int, arg.split(",")))


def select_device(gpu_idx=0, verbose=True):
    if verbose:
        print(
            "============================================================================================"
        )
        # set device to cpu or cuda
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        print(
            "============================================================================================"
        )
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
    return device


def get_args(verbose=True):
    """Call args"""
    parser = argparse.ArgumentParser()

    # WandB and Logging parameters
    parser.add_argument(
        "--project", type=str, default="FourRoom", help="WandB project classification"
    )
    parser.add_argument(
        "--logdir", type=str, default="log/train_log", help="name of the logging folder"
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Global folder name for experiments with multiple seed tests.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help='Seed-specific folder name in the "group" folder.',
    )
    parser.add_argument(
        "--algo-name",
        type=str,
        default="EigenOption",
        help="SNAC / EigenOption / CoveringOption / PPO",
    )
    parser.add_argument(
        "--log-interval", type=int, default=1, help="logging interval; epoch-based"
    )
    parser.add_argument(
        "--env-seed",
        type=int,
        default=0,
        help="Seed to fix the agent location",
    )
    parser.add_argument(
        "--seeds",
        type=list_of_ints,
        default=[1, 2, 3, 4, 5],  # 0, 2
        help="seeds for computational stochasticity --seeds 1,3,5,7,9 # without space",
    )

    # OpenAI Gym parameters
    parser.add_argument(
        "--env-name",
        type=str,
        default="FourRooms",
        help="This specifies which environment one is working with= FourRooms or CtF1v1, CtF1v2}",
    )
    parser.add_argument(
        "--SF-epoch",
        type=int,
        default=250,  # 500
        help="total number of epochs; every epoch it does evaluation",
    )
    parser.add_argument(
        "--PPO-epoch",
        type=int,
        default=500,  # 300
        help="For PPO alg. Total number of epochs; every epoch it does evaluation",
    )
    parser.add_argument(
        "--RP-epoch",
        type=int,
        default=0,
        help="This is not used. total number of epochs; every epoch it does evaluation",
    )
    parser.add_argument(
        "--OP-epoch",
        type=int,
        default=20,  # 50
        help="total number of epochs to train one each option policy; every epoch it does evaluation",
    )
    parser.add_argument(
        "--HC-epoch",
        type=int,
        default=300,  # 500
        help="total number of epochs; every epoch it does evaluation",
    )
    parser.add_argument(
        "--Psi-epoch",
        type=int,
        default=0,  # 10
        help="total number of epochs; every epoch it does evaluation",
    )
    parser.add_argument(
        "--step-per-epoch",
        type=int,
        default=200,  # 200
        help="number of iterations within one epoch",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        help="number of threads to use in sampling; \
                            sampler will select threads number with this limit",
    )
    parser.add_argument(
        "--episode-len",
        type=int,
        default=100,
        help="episodic length; useful when one wants to constrain to long to short horizon",
    )
    parser.add_argument(
        "--episode-num",
        type=int,
        default=6,
        help="number of episodes to collect for one env",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="number of episodes for evaluation; mean of those is returned as eval performance",
    )

    # some params
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1,
        help="Changing this requires redesign of CNN. tensor image size",
    )
    parser.add_argument(
        "--img-tile-size",
        type=int,
        default=32,
        help="32 is default. This is used for logging the images of training progresses. image tile size",
    )

    # network params
    parser.add_argument(
        "--feaNet-type",
        type=str,
        default="CNN",
        help="CNN or VAE",
    )

    # dimensional params
    parser.add_argument(
        "--a-dim",
        type=int,
        default=4,
        help="One can arbitrarily set the max dimension of action when one wants to disregard other useless action components of Minigrid",
    )

    parser.add_argument(
        "--fc-dim",
        type=int,
        default=128,
        help="This is general fully connected dimension for most of network this code.",
    )

    parser.add_argument(
        "--conv-fc-dim",
        type=int,
        default=1024,
        help="This is a dimension of FCL that decodes the output of CNN",
    )
    parser.add_argument(
        "--sf-dim",
        type=int,
        default=128,
        help="This is an feature dimension thus option dimension. 32 / 64",
    )
    parser.add_argument(
        "--num-vector",
        type=int,
        default=6,
        help="Must be divided by 2. ex) 10, 20, 30",
    )

    # learning rates
    parser.add_argument(
        "--policy-lr", type=float, default=1e-4, help="PPO-actor learning rate"
    )
    parser.add_argument(
        "--critic-lr", type=float, default=3e-4, help="PPO-critic learning rate"
    )
    parser.add_argument(
        "--feature-lr",
        type=float,
        default=2e-4,
        help="CNN lr where scheduler is used so can be high",
    )
    parser.add_argument(
        "--psi-lr",
        type=float,
        default=3e-4,
        help="Intermediate-level model learning rate",
    )
    parser.add_argument(
        "--option-lr",
        type=float,
        default=2e-4,
        help="Intermediate-level model learning rate",
    )

    # PPO parameters
    parser.add_argument(
        "--K-epochs", type=int, default=5, help="PPO update per one iter"
    )
    parser.add_argument(
        "--eps-clip", type=float, default=0.2, help="clipping parameter for gradient"
    )
    parser.add_argument(
        "--entropy-scaler",
        type=float,
        default=1e-4,
        help="entropy scaler from PPO action-distribution",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.95,
        help="Used in advantage estimation for numerical stability",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="discount parameters")
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=-0.5,
        help="min deviation as e^sig_min ~= 0.6",
    )
    parser.add_argument(
        "--sigma-max", type=float, default=0.5, help="max deviation as e^sig_max ~= 1.6"
    )

    # Training parameters
    parser.add_argument(
        "--max-num-traj",
        type=int,
        default=200,
        help="This sets the max number of trajectories the buffer will store. Exceeding will replace oldest trjs",
    )

    parser.add_argument(
        "--min-num-traj",
        type=int,
        default=100,
        help="For buffer learing, this sets the sub-iterations",
    )

    parser.add_argument(
        "--trj-per-iter",
        type=int,
        default=10,
        help="This sets the number of trajectories to use for one sub-iteration",
    )

    # Misc. parameters
    parser.add_argument(
        "--rendering",
        type=bool,
        default=True,
        help="saves the rendering during evaluation",
    )
    parser.add_argument(
        "--draw-map",
        type=bool,
        default=True,
        help="Turn off plotting reward map. Only works for FourRoom",
    )
    parser.add_argument(
        "--import-sf-model",
        type=bool,
        default=False,
        help="it imports previously trained model",
    )
    parser.add_argument(
        "--import-ppo-model",
        type=bool,
        default=False,
        help="it imports previously trained model",
    )
    parser.add_argument(
        "--import-op-model",
        type=bool,
        default=False,
        help="it imports previously trained model",
    )
    parser.add_argument(
        "--import-hc-model",
        type=bool,
        default=False,
        help="it imports previously trained model",
    )

    parser.add_argument("--gpu-idx", type=int, default=0, help="gpu idx to train")
    parser.add_argument("--verbose", type=bool, default=True, help="WandB logging")

    args = parser.parse_args()

    # post args processing
    args.device = select_device(args.gpu_idx, verbose)

    if args.import_op_model and not args.import_sf_model:
        print("\tWarning: importing OP model without Pre-trained SF")
    if (args.import_hc_model and not args.import_op_model) or (
        args.import_hc_model and not args.import_sf_model
    ):
        print("\tWarning: importing HC model without Pre-trained SF/OP")

    return args
