from utils.get_all_states import get_grid_tensor, get_grid_tensor2
from utils.call_env import call_env
from utils import *
import numpy as np


args = get_args(verbose=False)
env = call_env(args)
if args.env_name == "FourRooms":
    grid_tensor, coords, loc = get_grid_tensor(env, 0)
elif args.env_name == "LavaRooms":
    grid_tensor, coords, loc = get_grid_tensor(env, 0)
elif args.env_name == "CtF1v2":
    grid_tensor, coords, loc = get_grid_tensor2(env, 0)

print(args.env_name)
print(grid_tensor.shape)

print(np.sum(grid_tensor, axis=-1))
print(np.sum(grid_tensor, axis=-1).shape)
