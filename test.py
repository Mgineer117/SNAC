from utils.call_env import call_env
from utils import *
import numpy as np
import matplotlib.pyplot as plt


args = get_args(verbose=False)

args.env_name = "LavaRooms"
args.grid_type = 0
env = call_env(args)    

obs, _ = env.reset(seed=0)

# img = obs['observation']
# img = np.sum(img, axis=-1)
# img = (img - img.min()) / (img.max() - img.min())

img = env.render()

plt.axis('off')  # Turns off the axis (gridlines and ticks)
plt.imshow(img)
plt.tight_layout()
plt.savefig('env_image.png')

