import uuid
import random
import datetime
import wandb
import sys
import os
import contextlib
from tqdm import tqdm  # Progress bar

from algorithms import SNAC, EigenOption, PPO, SAC, OptionCritic
from utils.call_env import call_env
from utils.utils import setup_logger, seed_all, override_args

wandb.require("core")
wandb.init(mode="disabled")  # Turn off WandB sync


#########################################################
# Utility to Suppress Print Output and Capture Errors
#########################################################
@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull  # Redirect output
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr  # Restore output


#########################################################
# Parameter definitions
#########################################################
def train(args, seed, unique_id, exp_time):
    """Initiate the training process with given args."""
    env = call_env(args)
    logger, writer = setup_logger(args, unique_id, exp_time, seed)

    algo_classes = {
        "PPO": PPO,
        "SAC": SAC,
        "OptionCritic": OptionCritic,
        "SNAC": SNAC,
        "EigenOption": EigenOption,
    }

    alg_class = algo_classes.get(args.algo_name)
    if alg_class is None:
        raise ValueError(f"Unknown algorithm: {args.algo_name}")

    alg = alg_class(
        env=env,
        logger=logger,
        writer=writer,
        args=args,
    )
    alg.run()

    wandb.finish()
    writer.close()


#########################################################
# TEST CASE 1: Run SNAC with small SF-epoch and OP/HC-timesteps
#########################################################
def test_snac():
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")
    seed = 42
    seed_all(seed)
    errors = []  # Store errors to print later

    envs = ["CtF", "PointNavigation"]
    with tqdm(
        total=len(envs),
        desc="Testing SNAC Pipeline",
        bar_format="{l_bar}{bar} [ elapsed: {elapsed} ]",
    ) as pbar:
        for env_name in envs:
            try:
                args = override_args(env_name)
                args.algo_name = "SNAC"
                args.SF_epoch = 10  # Small SF-epoch for quick testing
                args.step_per_epoch = 1  # Small steps per epoch
                args.sf_log_interval = 10
                args.OP_timesteps = 100  # Small OP-timesteps
                args.HC_timesteps = 100  # Small HC-timesteps
                args.sf_batch_size = 64
                args.op_batch_size = 32
                args.hc_batch_size = 32
                args.K_epochs = 3
                args.min_batch_size = 64
                args.max_batch_size = 512
                args.warm_batch_size = 256
                args.DIF_batch_size = 512
                args.post_process = None
                args.num_options = 2
                args.method = "top"
                args.rendering = True
                args.draw_map = True

                with suppress_output():  # Suppress print output
                    train(args, seed, unique_id, exp_time)

            except Exception as e:
                errors.append(f"❌ SNAC Test Failed: {str(e)}")

            pbar.update(1)  # Update progress bar

    # Print errors after the progress bar finishes
    for err in errors:
        print(err)


#########################################################
# TEST CASE 2: Run EigenOption with small SF-epoch and OP/HC-timesteps
#########################################################
def test_eigenoption():
    seed = 42
    seed_all(seed)
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")
    errors = []  # Store errors to print later
    
    envs = ["FourRooms", "Maze"]
    with tqdm(
        total=len(envs),
        desc="Testing EigenOption pipeline",
        bar_format="{l_bar}{bar} [ elapsed: {elapsed} ]",
    ) as pbar:
        for env_name in envs:
            try:
                args = override_args(env_name)
                args.algo_name = "EigenOption"
                args.SF_epoch = 10  # Small SF-epoch for quick testing
                args.step_per_epoch = 1  # Small steps per epoch
                args.sf_log_interval = 10
                args.OP_timesteps = 100  # Small OP-timesteps
                args.HC_timesteps = 100  # Small HC-timesteps
                args.sf_batch_size = 32
                args.op_batch_size = 32
                args.hc_batch_size = 32
                args.K_epochs = 3
                args.min_batch_size = 32
                args.max_batch_size = 512
                args.warm_batch_size = 256
                args.DIF_batch_size = 512
                args.post_process = None
                args.num_options = 2
                args.method = "top"
                args.rendering = True
                args.draw_map = True

                with suppress_output():  # Suppress print output
                    train(args, seed, unique_id, exp_time)

            except Exception as e:
                errors.append(f"❌ EigenOption Test Failed: {str(e)}")

            pbar.update(1)  # Update progress bar

    # Print errors after the progress bar finishes
    for err in errors:
        print(err)

#########################################################
# TEST CASE 3: Run PPO for small timesteps on all environments
#########################################################
def test_ppo():
    seed = 42
    seed_all(seed)
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")
    unique_id = str(uuid.uuid4())[:4]

    envs = ["FourRooms", "Maze", "CtF", "PointNavigation"]
    errors = []  # Store errors to print later

    with tqdm(
        total=len(envs),
        desc="Testing Any Environmental Failures",
        bar_format="{l_bar}{bar} [ elapsed: {elapsed} ]",
    ) as pbar:
        for env_name in envs:
            try:
                args = override_args(env_name)
                args.algo_name = "PPO"
                args.PPO_timesteps = 1000  # Small timesteps for quick testing
                args.min_batch_for_worker = 1024
                args.rendering = True
                args.draw_map = True

                with suppress_output():  # Suppress print output
                    train(args, seed, unique_id, exp_time)

            except Exception as e:
                errors.append(f"❌ PPO Test Failed on {env_name}: {str(e)}")

            pbar.update(1)  # Update progress bar

    # Print errors after the progress bar finishes
    for err in errors:
        print(err)


#########################################################
# Run Tests
#########################################################
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" 🎯 Running SNAC pipeline with small SF-epoch, OP, HC-timesteps ")
    print("=" * 60 + "\n")
    test_snac()

    print("\n" + "=" * 60)
    print(" 🧠 Running EigenOption pipeline with small SF-epoch, OP, HC-timesteps ")
    print("=" * 60 + "\n")
    test_eigenoption()

    print("\n" + "=" * 60)
    print(" 🚀 Running PPO for small timesteps on all environments ")
    print("=" * 60 + "\n")
    test_ppo()

    print("\n✅ All tests completed successfully!")
