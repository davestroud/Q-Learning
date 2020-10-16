import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T




env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# A named tuple representing a single transition in our enviroment. 
# It maps (state, action) pairs to their (next_state,reward) result
# with the state being the screen difference image.
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

