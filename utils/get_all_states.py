import os
import numpy as np
import gymnasium as gym
from tqdm import trange
import matplotlib.pyplot as plt
from utils.wrappers import NoStateDictWrapper
from minigrid.core.grid import Grid


def get_grid_tensor(env, env_seed):
    """
    Can be extended to the multigrid by removing multiple agents
    FourRoom and LavaRoom tailored
    """
    obs, _ = env.reset(seed=env_seed)
    grid_tensor = obs["observation"]

    loc = np.where(grid_tensor[:, :, 0] == 10)
    grid_tensor[loc[0], loc[1], 0] = 1
    env.close()

    x_coords, y_coords = np.where(
        (grid_tensor[:, :, 0] != 2)
        & (grid_tensor[:, :, 0] != 4)
        & (grid_tensor[:, :, 0] != 8)
    )  # find idx where not wall

    return grid_tensor, (x_coords, y_coords), loc


def get_grid_tensor2(env, env_seed):
    """
    Can be extended to the multigrid by removing multiple agents
    CTF tailored
    """
    obs, _ = env.reset(seed=env_seed)
    grid_tensor = obs["observation"]

    loc = np.where(grid_tensor[:, :, 1] == 1)
    grid_tensor[loc[0], loc[1], 1] = 0
    grid_tensor[loc[0], loc[1], 2] = 0
    env.close()

    x_coords, y_coords = np.where(
        (grid_tensor[:, :, 0] != 0)
        & (grid_tensor[:, :, 1] != 2)
        & (grid_tensor[:, :, 1] != 3)
        & (grid_tensor[:, :, 1] != 4)
    )  # find idx where not wall

    return grid_tensor, (x_coords, y_coords), loc


def generate_possible_states(env, path, args):
    # Check if the given render mode is not human
    if env.render_mode != "rgb_array":
        raise ValueError(f"render mode should be rgb_array. Current: {env.render_mode}")

    # convert to fully observable wrapper which returns the encoded tensors
    full_env = NoStateDictWrapper(env)

    # get the raw grid tensor without agent in the image
    grid_tensor = get_grid_tensor(full_env)
    # get coordinates where the agent can visit
    x_coords, y_coords = np.where(grid_tensor[:, :, 0] != 2)  # find idx where not wall

    # call a grid class with the given grid tensor
    grid, _ = Grid.decode(grid_tensor)

    # there are four possible directions
    # this should be changed in another environmental domains
    dirs = [0, 1, 2, 3]

    # create a path for saving each directional map separately
    path = os.path.join(path, "allStates")
    os.mkdir(path)
    for dir in dirs:
        temp_path = os.path.join(path, str(dir))
        os.mkdir(temp_path)

    total_iterations = len(dirs) * len(x_coords)  # Total number of iterations
    idx = 0
    # Progress bar with trange
    with trange(total_iterations, desc="Generating all possible state images") as pbar:
        for dir in dirs:
            for x, y in zip(x_coords, y_coords):
                # Render the new state
                agent_pos = (x, y)
                img = grid.render(
                    tile_size=args.tile_size,
                    agent_pos=agent_pos,
                    agent_dir=dir,
                    highlight_mask=None,
                )

                # Save the image
                plt.imshow(img)
                plt.savefig(os.path.join(path, str(dir), f"{idx}.png"))
                plt.close()
                idx += 1
                pbar.update(1)  # Update the progress bar by 1 for each iteration

    args.path_allStates = path
    return grid_tensor, (x_coords, y_coords)


def generate_possible_tensors(env, path, args, tile_size, env_seed):
    # Check if the given render mode is not human
    if env.render_mode != "rgb_array":
        raise ValueError(f"render mode should be rgb_array. Current: {env.render_mode}")

    # get the raw grid tensor without agent in the image
    grid_tensor, original_tensor = get_grid_tensor(env, env_seed)
    # get coordinates where the agent can visit
    x_coords, y_coords = np.where(
        grid_tensor[:, :, 0] != 2 and grid_tensor[:, :, 0] != 8
    )  # find idx where not wall

    # there are four possible directions
    # this should be changed in another environmental domains

    # create a path for saving each directional map separately
    path = os.path.join(path, "allStates")
    os.mkdir(path)

    total_iterations = len(x_coords)  # Total number of iterations

    # Progress bar with trange
    with trange(total_iterations, desc="Generating all possible state images") as pbar:
        for x, y in zip(x_coords, y_coords):
            # Render the new state
            temp_img = grid_tensor.copy()
            temp_img[x, y, 0] = 10
            img = temp_img.copy()  # * 20
            img = np.repeat(np.repeat(img, tile_size, axis=0), tile_size, axis=1)
            img = np.squeeze(img)  # (n, n) 2d image for grey scale saving

            del temp_img

            # Save the image
            plt.imsave(
                os.path.join(path, f"{str(x)}_{str(y)}.png"),
                img,
                cmap="gray",
            )
            plt.close()
            pbar.update(1)  # Update the progress bar by 1 for each iteration

    args.path_allStates = path
    return original_tensor, (x_coords, y_coords)
