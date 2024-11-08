import numpy as np
from numpy.typing import NDArray
import gymnasium as gym


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


class NoStateDictWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, tile_size: int = 1):
        super(NoStateDictWrapper, self).__init__(env)
        self.tile_size = tile_size

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = observation['image']
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
        observation = observation['image']
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
        obs = {}
        obs["observation"] = observation

        agent_pos = self.get_wrapper_attr("agents")[0].pos
        obs["agent_pos"] = np.array(agent_pos)
        return obs, {}

    def step(self, action):
        # Call the original step method
        observation, reward, termination, truncation, info = self.env.step(action)
        # observation = observation.reshape(-1, 1, 1)
        observation = np.repeat(
            np.repeat(observation, self.tile_size, axis=0), self.tile_size, axis=1
        )
        obs = {}
        obs["observation"] = observation

        agent_pos = self.get_wrapper_attr("agents")[0].pos
        obs["agent_pos"] = np.array(agent_pos)
        return obs, reward, termination, truncation, info
