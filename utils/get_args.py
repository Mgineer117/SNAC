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

    ### Adjustable parameters

    ### WandB and Logging parameters
    parser.add_argument(
        "--project", type=str, default="Test", help="WandB project classification"
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
        "--sf-log-interval",
        type=int,
        default=None,
        help="logging interval; epoch-based",
    )
    parser.add_argument(
        "--op-log-interval",
        type=int,
        default=None,
        help="logging interval; epoch-based",
    )
    parser.add_argument(
        "--hc-log-interval",
        type=int,
        default=None,
        help="logging interval; epoch-based",
    )
    parser.add_argument(
        "--oc-log-interval",
        type=int,
        default=None,
        help="logging interval; epoch-based",
    )
    parser.add_argument(
        "--ppo-log-interval",
        type=int,
        default=None,
        help="logging interval; epoch-based",
    )
    parser.add_argument(
        "--sac-log-interval",
        type=int,
        default=None,
        help="logging interval; epoch-based",
    )

    ### Environmental / Running parameters
    parser.add_argument(
        "--env-name",
        type=str,
        default="FourRooms",
        help="This specifies which environment one is working with= FourRooms or CtF1v1, CtF1v2}",
    )
    parser.add_argument(
        "--ctf-map",
        type=str,
        default=None,
        help="This specifies which environment one is working with= FourRooms or CtF1v1, CtF1v2}",
    )
    parser.add_argument(
        "--algo-name",
        type=str,
        default="SNAC",
        help="SNAC / EigenOption / CoveringOption / PPO",
    )
    parser.add_argument(
        "--grid-type",
        type=int,
        default=0,
        help="0 or 1. Seed to fix the grid, agent, and goal locations",
    )
    parser.add_argument(
        "--episode-len",
        type=int,
        default=None,
        help="episodic length; useful when one wants to constrain to long to short horizon",
    )
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
    parser.add_argument(
        "--cost-scaler",
        type=float,
        default=1e-0,
        help="reward shaping parameter r = reawrd - scaler * cost",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="number of episodes for evaluation; mean of those is returned as eval performance",
    )
    parser.add_argument(
        "--post-process",
        type=str,
        default=None,
        help="number of episodes for evaluation; mean of those is returned as eval performance",
    )
    parser.add_argument(
        "--seeds",
        type=list_of_ints,
        default=[1, 2, 3, 4, 5],  # 0, 2
        help="seeds for computational stochasticity --seeds 1,3,5,7,9 # without space",
    )

    ### Algorithmic iterations
    parser.add_argument(
        "--SF-epoch",
        type=int,
        default=None,  # 1000
        help="total number of epochs for SFs training",
    )
    parser.add_argument(
        "--Psi-epoch",
        type=int,
        default=None,  # 10
        help="total number of epochs for Psi network training. If 0>, Q = Psi*w action is used instead.",
    )
    parser.add_argument(
        "--OP-epoch",
        type=int,
        default=None,  # 500
        help="total number of epochs for OP training",
    )
    parser.add_argument(
        "--HC-epoch",
        type=int,
        default=None,  # 500
        help="total number of epochs for HC training",
    )
    parser.add_argument(
        "--OC-epoch",
        type=int,
        default=None,  # 500
        help="total number of epochs for OC training",
    )
    parser.add_argument(
        "--PPO-epoch",
        type=int,
        default=None,  # 500
        help="total number of epochs for OC training",
    )
    parser.add_argument(
        "--SAC-epoch",
        type=int,
        default=None,  # 500
        help="total number of epochs for SAC training",
    )
    parser.add_argument(
        "--step-per-epoch",
        type=int,
        default=None,  # 10
        help="number of iterations within one epoch",
    )
    parser.add_argument(
        "--bfgs-iter",
        type=int,
        default=10,
        help="Number of bfgs iterations for one minibatch",
    )

    ### Learning rates
    parser.add_argument(
        "--feature-lr",
        type=float,
        default=None,
        help="SFs train lr where scheduler is used so can be high",
    )
    parser.add_argument(
        "--option-lr",
        type=float,
        default=None,
        help="option vector lr",
    )
    parser.add_argument(
        "--psi-lr",
        type=float,
        default=3e-4,
        help="psi network lr",
    )
    parser.add_argument(
        "--op-policy-lr", type=float, default=1e-3, help="Option network lr"
    )
    parser.add_argument(
        "--op-critic-lr",
        type=float,
        default=None,
        help="Option policy (PPO-based) critic learning rate. If none, BFGS is used.",
    )
    parser.add_argument(
        "--hc-policy-lr",
        type=float,
        default=3e-4,
        help="Hierarchical Controller network lr",
    )
    parser.add_argument(
        "--hc-critic-lr",
        type=float,
        default=None,
        help="Hierarchical Policy policy (PPO-based) critic learning rate. If none, BFGS is used.",
    )
    parser.add_argument(
        "--ppo-policy-lr", type=float, default=3e-4, help="PPO-actor learning rate"
    )
    parser.add_argument(
        "--ppo-critic-lr",
        type=float,
        default=None,
        help="PPO-critic learning rate. If none, BFGS is used.",
    )

    ### Algorithmic parameters
    parser.add_argument(
        "--op-mode",
        type=str,
        default="ppo",
        help="ppo / sac. Either algorithm is used for option network training.",
    )
    parser.add_argument(
        "--obs-norm",
        type=str,
        default="none",
        help="ema / cma. ema: Explonential / Cumulative moving average. Observation normalization for each network.",
    )
    parser.add_argument(
        "--min-option-length",
        type=int,
        default=5,
        help="Minimum time step for one option duration of SNAC / EigenOption",
    )
    parser.add_argument(
        "--min-cover-option-length",
        type=int,
        default=25,
        help="Minimum time step for one option duration of successive option",
    )
    parser.add_argument(
        "--num-vector",
        type=int,
        default=16,
        help="Must be divided by 4. ex) 8, 12, 16. Minimum = 8.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Base number of batch size for training",
    )
    parser.add_argument(
        "--op-batch-size",
        type=int,
        default=None,
        help="Option policy number of batch size for training",
    )
    parser.add_argument(
        "--hc-batch-size",
        type=int,
        default=None,
        help="Hierarchical policy number of batch size for training",
    )
    parser.add_argument(
        "--ppo-batch-size",
        type=int,
        default=None,
        help="Naive ppo number of batch size for training",
    )
    parser.add_argument(
        "--sac-batch-size",
        type=int,
        default=None,
        help="SAC number of batch size for training",
    )
    parser.add_argument(
        "--oc-batch-size",
        type=int,
        default=None,
        help="Option critic number of batch size for training",
    )
    parser.add_argument(
        "--min-batch-for-worker",
        type=int,
        default=1024,
        help="Minimum batch size assgined for one worker (thread)",
    )
    parser.add_argument(
        "--op-entropy-scaler",
        type=float,
        default=5e-3,
        help="Option policy entropy scaler",
    )
    parser.add_argument(
        "--hc-entropy-scaler",
        type=float,
        default=5e-2,
        help="Hierarchical policy entropy scaler",
    )
    parser.add_argument(
        "--ppo-entropy-scaler",
        type=float,
        default=5e-2,
        help="PPO policy entropy scaler",
    )
    parser.add_argument(
        "--PM-policy",
        type=str,
        default=None,
        help="RW / PPO. The choice of primitive policy for hierarchical policy.",
    )

    ### SF param (loss scale)
    parser.add_argument(
        "--phi-loss-r-scaler",
        type=float,
        default=None,
        help="Scaler to SFs reward regression loss",
    )
    parser.add_argument(
        "--phi-loss-s-scaler",
        type=float,
        default=None,
        help="Scaler to SFs latent state regression loss",
    )
    parser.add_argument(
        "--phi-loss-kl-scaler",
        type=float,
        default=None,
        help="Scaler to SFs latent state regression loss (VAE only)",
    )
    parser.add_argument(
        "--phi-loss-l2-scaler",
        type=float,
        default=None,
        help="Scaler to SFs network weight to prevent overfitting",
    )
    parser.add_argument(
        "--num-traj-decomp",
        type=int,
        default=None,
        help="Number of trajectory to decompose via SVD for option discovery",
    )
    parser.add_argument(
        "--max-num-traj",
        type=int,
        default=None,
        help="Maximum number of trajectories the buffer can store. Exceeding it will refresh the oldest trajectory",
    )
    parser.add_argument(
        "--min-num-traj",
        type=int,
        default=None,
        help="Minimum number of trajectory to start training.",
    )
    parser.add_argument(
        "--trj-per-iter",
        type=int,
        default=None,
        help="Number of trajectory to pull out from the buffer to train SFs",
    )

    ### Resorces
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        help="Number of threads to use in sampling. If none, sampler will select available threads number with this limit",
    )
    parser.add_argument(
        "--cpu-preserve-rate",
        type=float,
        default=0.95,
        help="For multiple run of experiments, one can set this to restrict the cpu threads the one exp uses for sampling.",
    )

    ### Dimensional params
    parser.add_argument(
        "--a-dim",
        type=int,
        default=None,
        help="action dimension. For grid with 5 available actions, it is one-hotted to be 1 x 5.",
    )
    parser.add_argument(
        "--fc-dim",
        type=int,
        default=None,
        help="This is general fully connected dimension for most of network this code.",
    )
    parser.add_argument(
        "--feature-fc-dim",
        type=int,
        default=None,
        help="This is a dimension of FCL that decodes the output of CNN or VAE",
    )
    parser.add_argument(
        "--option-fc-dim",
        type=int,
        default=None,
        help="This is a dimension of FCL of option policy",
    )
    parser.add_argument(
        "--sf-dim",
        type=int,
        default=None,
        help="This is an latent feature dimension thus option dimension.",
    )

    # PPO parameters
    parser.add_argument(
        "--K-epochs", type=int, default=10, help="PPO update per one iter"
    )
    parser.add_argument(
        "--OP-K-epochs", type=int, default=15, help="Option policy update per one iter"
    )
    parser.add_argument(
        "--eps-clip", type=float, default=0.2, help="clipping parameter for gradient"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.95,
        help="Used in advantage estimation.",
    )

    # SAC parameters
    parser.add_argument(
        "--tune-alpha", type=bool, default=True, help="Automatic entropy scaler."
    )
    parser.add_argument(
        "--sac-policy-lr", type=float, default=3e-4, help="SAC-actor learning rate"
    )
    parser.add_argument(
        "--sac-critic-lr",
        type=float,
        default=1e-4,
        help="SAC-critic learning rate. (AdamW)",
    )
    parser.add_argument(
        "--sac-alpha-lr",
        type=float,
        default=1e-4,
        help="Lr for auto-tune entropy scaler",
    )
    parser.add_argument(
        "--sac-init-alpha",
        type=float,
        default=0.2,
        help="Initial entropy scaler",
    )
    parser.add_argument(
        "--sac-soft-update-rate",
        type=float,
        default=0.005,
        help="Target critic network update. Lower the slower rate of update",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1,
        help="Interval to perform target critic update in SAC",
    )
    parser.add_argument(
        "--sac-max-num-traj",
        type=int,
        default=10000,
        help="Max number of trajectory the buffer will contain in SAC",
    )
    parser.add_argument(
        "--sac-min-num-traj",
        type=int,
        default=100,
        help="Min number of trajectory the buffer will contain in SAC",
    )
    parser.add_argument(
        "--sac-trj-per-iter",
        type=int,
        default=25,
        help="N",
    )
    parser.add_argument(
        "--sac-step-per-epoch",
        type=int,
        default=None,
        help="This sets the number of trajectories to use for one sub-iteration",
    )
    parser.add_argument(
        "--sac-entropy-scaler",
        type=float,
        default=5e-2,
        help="entropy scaler from PPO action-distribution",
    )

    parser.add_argument("--gamma", type=float, default=None, help="discount parameters")

    # Misc. parameters
    parser.add_argument(
        "--rendering",
        type=bool,
        default=False,
        help="saves the rendering during evaluation",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=None,
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
        default=True,
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
    parser.add_argument(
        "--import-sac-model",
        type=bool,
        default=False,
        help="it imports previously trained model",
    )
    parser.add_argument(
        "--import-oc-model",
        type=bool,
        default=False,
        help="it imports previously trained model",
    )

    parser.add_argument("--gpu-idx", type=int, default=0, help="gpu idx to train")
    parser.add_argument("--verbose", type=bool, default=False, help="WandB logging")

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
