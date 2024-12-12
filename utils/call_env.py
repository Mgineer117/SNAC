import safety_gymnasium as sgym
import gymnasium as gym

from safety_gymnasium import __register_helper
from gym_multigrid.envs.fourrooms import FourRooms
from gym_multigrid.envs.maze import Maze
from gym_multigrid.envs.lavarooms import LavaRooms
from gym_multigrid.envs.ctf import CtF


from utils.wrappers import GridWrapper, CtFWrapper, NavigationWrapper, GymWrapper


def disc_or_cont(env, args):
    if isinstance(env.action_space, gym.spaces.Discrete):
        args.is_discrete = True
        print(f"Discrete Action Space")
    elif isinstance(env.action_space, gym.spaces.Box):
        args.is_discrete = False
        print(f"Continuous Action Space")
    else:
        raise ValueError(f"Unknown action space type {env.action_space}.")


def call_env(args):
    # define the env
    if args.env_name == "FourRooms":
        # first call dummy env to find possible location for agent
        env = FourRooms(
            grid_type=args.grid_type,
            max_steps=args.episode_len,
            tile_size=args.img_tile_size,
            highlight_visible_cells=False,
            partial_observability=False,
            render_mode="rgb_array",
        )
        disc_or_cont(env, args)
        args.agent_num = len(env.agents)
        return GridWrapper(env, tile_size=args.tile_size)
    elif args.env_name == "Maze":
        # first call dummy env to find possible location for agent
        env = Maze(
            grid_type=args.grid_type,
            max_steps=args.episode_len,
            tile_size=args.img_tile_size,
            highlight_visible_cells=False,
            partial_observability=False,
            render_mode="rgb_array",
        )
        disc_or_cont(env, args)
        args.agent_num = len(env.agents)
        return GridWrapper(env, tile_size=args.tile_size)
    elif args.env_name == "LavaRooms":
        # first call dummy env to find possible location for agent
        env = LavaRooms(
            grid_type=args.grid_type,
            max_steps=args.episode_len,
            tile_size=args.img_tile_size,
            highlight_visible_cells=False,
            partial_observability=False,
            render_mode="rgb_array",
        )
        disc_or_cont(env, args)
        args.agent_num = len(env.agents)
        return GridWrapper(env, tile_size=args.tile_size)

    elif args.env_name in ("CtF1v1", "CtF1v2", "CtF1v3", "CtF1v4"):
        map_path: str = "assets/ctf_avoid_obj.txt"
        observation_option: str = "tensor"
        env_name = args.env_name
        red_agents = int(
            env_name.split("v")[1]
        )  # Extract the number of red agents from the env name
        if env_name.startswith("CtF1v"):
            env = CtF(
                map_path=map_path,
                num_blue_agents=1,
                num_red_agents=red_agents,
                observation_option=observation_option,
                step_penalty_ratio=0.0,
            )
        else:
            raise NotImplementedError(f"{args.env_name} not implemented")
        disc_or_cont(env, args)
        args.agent_num = len(env.agents)
        return CtFWrapper(env, tile_size=args.tile_size)
    elif args.env_name == "PointNavigation":
        config = {"agent_name": "Point"}
        env_id = "PointNavigation"
        __register_helper(
            env_id=env_id,
            entry_point="gym_continuous.env_builder:Builder",
            spec_kwargs={"config": config, "task_id": env_id},
            max_episode_steps=args.episode_len,
        )

        env = sgym.make(
            "PointNavigation",
            render_mode="rgb_array",
            width=1024,
            height=1024,
            camera_name="fixedfar",
        )

        disc_or_cont(env, args)
        args.agent_num = 1
        return NavigationWrapper(env)
    elif args.env_name == "InvertedPendulum":
        env = gym.make(
            "InvertedPendulum-v4",
            render_mode="rgb_array",
        )
        disc_or_cont(env, args)
        args.agent_num = 1
        return GymWrapper(env)
    elif args.env_name == "Hopper":
        env = gym.make(
            "Hopper-v4",
            render_mode="rgb_array",
        )
        disc_or_cont(env, args)
        args.agent_num = 1
        return GymWrapper(env)
    else:
        raise ValueError(f"Invalid environment key: {args.env_name}")
