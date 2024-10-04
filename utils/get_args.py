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
        "--project", type=str, default="4ROOM", help="WandB project classification"
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
        default="PPO",
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
    parser.add_argument("--env-name", type=str, default="FourRooms", help="ss")
    parser.add_argument(
        "--SF-epoch",
        type=int,
        default=250,  # 200
        help="total number of epochs; every epoch it does evaluation",
    )
    parser.add_argument(
        "--PPO-epoch",
        type=int,
        default=50,  # 50
        help="total number of epochs; every epoch it does evaluation",
    )
    parser.add_argument(
        "--RP-epoch",
        type=int,
        default=0,
        help="total number of epochs; every epoch it does evaluation",
    )
    parser.add_argument(
        "--OP-epoch",
        type=int,
        default=20,  # 25
        help="total number of epochs; every epoch it does evaluation",
    )
    parser.add_argument(
        "--HC-epoch",
        type=int,
        default=20,  # 50
        help="total number of epochs; every epoch it does evaluation",
    )
    parser.add_argument(
        "--Psi-epoch",
        type=int,
        default=10,  # 20
        help="total number of epochs; every epoch it does evaluation",
    )
    parser.add_argument(
        "--step-per-epoch",
        type=int,
        default=200,
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
    parser.add_argument("--tile-size", type=int, default=1, help="tensor image size")
    parser.add_argument("--img-tile-size", type=int, default=32, help="image tile size")
    parser.add_argument("--a-dim", type=int, default=4)

    # general params
    parser.add_argument("--fc-dim", type=int, default=64)

    # SF Network parameters
    parser.add_argument("--conv-fc-dim", type=int, default=256)
    parser.add_argument("--sf-dim", type=int, default=128)

    parser.add_argument("--policy_fc_hidden_dims", type=tuple, default=(64, 64))
    parser.add_argument(
        "--policy-output-dim",
        type=int,
        default=4,
        help="This constrains the max action output",
    )

    parser.add_argument(
        "--num-vector",
        type=int,
        default=10,
        help="Must be divided by 2. ex) 10, 20, 30",
    )

    # learning rates
    parser.add_argument(
        "--policy-lr", type=float, default=3e-4, help="PPO-actor learning rate"
    )
    parser.add_argument(
        "--critic-lr", type=float, default=5e-4, help="PPO-critic learning rate"
    )
    parser.add_argument(
        "--feature-lr",
        type=float,
        default=3e-4,
        help="Intermediate-level model learning rate",
    )
    parser.add_argument(
        "--psi-lr",
        type=float,
        default=5e-4,
        help="Intermediate-level model learning rate",
    )
    parser.add_argument(
        "--option-lr",
        type=float,
        default=1e-4,
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
        default=1e-3,
        help="entropy scaler from PPO action-distribution",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.95,
        help="Used in advantage estimation for numerical stability",
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="discount parameters")
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
        "--num-traj",
        type=int,
        default=30,
        help="embedding dimension both for categorical network and VAE",
    )

    parser.add_argument(
        "--update-iter",
        type=int,
        default=3,
        help="embedding dimension both for categorical network and VAE",
    )

    parser.add_argument(
        "--trj-per-iter",
        type=int,
        default=5,
        help="embedding dimension both for categorical network and VAE",
    )

    # Algorithmic parameters
    parser.add_argument(
        "--normalize-state", type=bool, default=False, help="normalise state input"
    )
    parser.add_argument(
        "--normalize-reward", type=bool, default=False, help="normalise reward input"
    )
    parser.add_argument(
        "--reward-scaler", type=float, default=None, help="reward scaler"
    )
    parser.add_argument(
        "--rendering",
        type=bool,
        default=True,
        help="saves the rendering during evaluation",
    )
    parser.add_argument(
        "--import-sf-model",
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
    parser.add_argument(
        "--import-ppo-model",
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
