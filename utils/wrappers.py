import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.core import ObservationWrapper
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX


class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding instead of the agent view.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FullyObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['image'].shape
        (7, 7, 3)
        >>> env_obs = FullyObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (11, 11, 3)
    """

    def __init__(self, env):
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.env.get_wrapper_attr("width"),
                self.env.get_wrapper_attr("height"),
                3,
            ),  # number of cells
            dtype="uint8",
        )

        self.observation_space = spaces.Dict({"image": new_image_space})
        # self.observation_space = spaces.Dict(
        #     {**self.observation_space.spaces, "image": new_image_space}
        # )

    def observation(self):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["blue"], env.agent_dir]
        )

        return {"image": full_grid}


class StateImageWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, args):
        super(StateImageWrapper, self).__init__(env)
        env.reset()
        image = env.render()
        s_dim = image.shape  # (width, height, colors)
        if s_dim[0] != s_dim[1] or len(s_dim) != 3:
            raise ValueError(
                f"This is not a square image: {s_dim[0] != s_dim[1]} or an image: {len(s_dim) != 3}"
            )

        action_space = env.action_space
        a_dim = action_space.shape

        ## should be manually chosen since there is dummy action dims
        args.s_dim = s_dim
        if args.a_dim is None:
            args.a_dim = a_dim
        env.close()

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        s = self.env.render()
        return s, info

    def step(self, action):
        # Call the original step method
        _, reward, termination, truncation, info = self.env.step(action)
        observation = self.env.render()
        observation = observation / 255.0  # image normalization
        return observation, reward, termination, truncation, info


class StateTensorWrapper(FullyObsWrapper):
    def __init__(self, env: gym.Env, args=None, tile_size: int = 1):
        super(StateTensorWrapper, self).__init__(env)
        self.tile_size = tile_size

        if args is not None:
            tensor, _ = env.reset()
            tensor = self.observation()
            tensor_image = tensor["image"]

            s_dim = tensor_image.shape  # (width, height, colors)
            if s_dim[0] != s_dim[1] or len(s_dim) != 3:
                raise ValueError(
                    f"This is not a square image: {s_dim[0] != s_dim[1]} or an image: {len(s_dim) != 3}"
                )

            action_space = env.action_space
            a_dim = action_space.shape

            ## should be manually chosen since there is dummy action dims
            args.s_dim = s_dim
            if args.a_dim is None:
                args.a_dim = a_dim
            env.close()

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        observation = self.observation()
        observation = observation["image"]

        observation = np.repeat(
            np.repeat(observation, self.tile_size, axis=0), self.tile_size, axis=1
        )

        return observation, {}

    def step(self, action):
        # Call the original step method
        _, reward, termination, truncation, info = self.env.step(action)
        observation = self.observation()
        observation = observation["image"]
        observation = np.repeat(
            np.repeat(observation, self.tile_size, axis=0), self.tile_size, axis=1
        )

        return observation, reward, termination, truncation, info


class NoStateDictWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, tile_size: int = 1):
        super(NoStateDictWrapper, self).__init__(env)
        self.tile_size = tile_size

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = observation["image"][:, :, 0:1]
        observation = np.repeat(
            np.repeat(observation, self.tile_size, axis=0), self.tile_size, axis=1
        )
        obs = {}
        obs["observation"] = observation

        agent_pos = self.get_wrapper_attr("agent_pos")
        obs["agent_pos"] = np.array(agent_pos)
        return obs, info

    def step(self, action):
        # Call the original step method
        observation, reward, termination, truncation, info = self.env.step(action)
        observation = observation["image"][:, :, 0:1]
        observation = np.repeat(
            np.repeat(observation, self.tile_size, axis=0), self.tile_size, axis=1
        )
        obs = {}
        obs["observation"] = observation

        agent_pos = self.get_wrapper_attr("agent_pos")
        obs["agent_pos"] = np.array(agent_pos)
        return obs, reward, termination, truncation, info


class NoStateDictCtfWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, tile_size: int = 1):
        super().__init__(env)
        self.tile_size = tile_size

    def reset(self, **kwargs):
        observation: NDArray
        observation, _ = self.env.reset(**kwargs)
        # observation = observation.reshape(-1, 1, 1)
        observation = np.repeat(
            np.repeat(observation, self.tile_size, axis=0), self.tile_size, axis=1
        )
        return observation, {}

    def step(self, action):
        # Call the original step method
        observation, reward, termination, truncation, info = self.env.step(action)
        # observation = observation.reshape(-1, 1, 1)
        observation = np.repeat(
            np.repeat(observation, self.tile_size, axis=0), self.tile_size, axis=1
        )
        return observation, reward, termination, truncation, info
