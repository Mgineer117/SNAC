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


class GridWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, tile_size: int = 1):
        super(GridWrapper, self).__init__(env)
        self.tile_size = tile_size
        self.agent_num = len(self.env.agents)

    def get_agent_pos(self):
        agent_pos = np.full((2 * self.agent_num,), np.nan, dtype=np.float32)
        for i in range(self.agent_num):
            agent_pos[2 * i : 2 * i + 2] = self.env.agents[i].pos
        return agent_pos

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation = observation["image"]
        observation = np.repeat(
            np.repeat(observation, self.tile_size, axis=0), self.tile_size, axis=1
        )
        obs = {}
        obs["observation"] = observation
        obs["agent_pos"] = self.get_agent_pos()
        return obs, info

    def step(self, action):
        action = np.argmax(action)
        # Call the original step method
        observation, reward, termination, truncation, info = self.env.step(action)
        observation = observation["image"]
        observation = np.repeat(
            np.repeat(observation, self.tile_size, axis=0), self.tile_size, axis=1
        )
        obs = {}
        obs["observation"] = observation
        obs["agent_pos"] = self.get_agent_pos()
        return obs, reward, termination, truncation, info


class CtFWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, tile_size: int = 1):
        super(CtFWrapper, self).__init__(env)
        self.tile_size = tile_size
        self.agent_num = len(self.env.agents)

    def get_agent_pos(self):
        agent_pos = np.full((2 * self.agent_num,), np.nan, dtype=np.float32)
        for i in range(self.agent_num):
            agent_pos[2 * i : 2 * i + 2] = self.env.agents[i].pos
        return agent_pos

    def reset(self, **kwargs):
        observation, _ = self.env.reset(**kwargs)
        observation = np.repeat(
            np.repeat(observation, self.tile_size, axis=0), self.tile_size, axis=1
        )
        obs = {}
        obs["observation"] = observation
        obs["agent_pos"] = self.get_agent_pos()
        return obs, {}

    def step(self, action):
        action = np.argmax(action)
        # Call the original step method
        observation, reward, termination, truncation, info = self.env.step(action)
        observation = np.repeat(
            np.repeat(observation, self.tile_size, axis=0), self.tile_size, axis=1
        )
        obs = {}
        obs["observation"] = observation
        obs["agent_pos"] = self.get_agent_pos()

        return obs, reward, termination, truncation, info


class NavigationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, cost_scaler: float = 1e-1):
        super(NavigationWrapper, self).__init__(env)
        self.cost_scaler = cost_scaler

    def get_agent_pos(self):
        return np.array([0, 0])

    def reset(self, **kwargs):
        observation, _ = self.env.reset(**kwargs)
        obs = {}
        obs["observation"] = observation
        obs["agent_pos"] = self.get_agent_pos()
        return obs, {}

    def step(self, action):
        # Call the original step method
        observation, reward, cost, termination, truncation, info = self.env.step(action)

        obs = {}
        obs["observation"] = observation
        obs["agent_pos"] = self.get_agent_pos()

        reward -= self.cost_scaler * cost

        if info["goal_met"]:
            truncation = True

        return obs, reward, termination, truncation, info


class GymWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(GymWrapper, self).__init__(env)

    def get_agent_pos(self):
        return np.array([0, 0])

    def reset(self, **kwargs):
        observation, _ = self.env.reset(**kwargs)
        obs = {}
        obs["observation"] = observation
        obs["agent_pos"] = self.get_agent_pos()
        return obs, {}

    def step(self, action):
        # Call the original step method
        observation, reward, termination, truncation, info = self.env.step(action)

        obs = {}
        obs["observation"] = observation
        obs["agent_pos"] = self.get_agent_pos()

        return obs, reward, termination, truncation, info
