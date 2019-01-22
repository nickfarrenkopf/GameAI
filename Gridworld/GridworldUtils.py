import random
import numpy as np

import paths
from learning.environments import GW1
from learning.environments import GW2
from learning.environments import GW3
from learning.environments import GW4
from learning.environments import GW5


### LOAD ###

def get_gridworld(num):
    """ """
    if num == 1:
        gw = GW1.Gridworld_1(paths)
    elif num == 2:
        gw = GW2.Gridworld_2(paths)
    elif num == 3:
        gw = GW3.Gridworld_3(paths)
    elif num == 4:
        gw = GW4.Gridworld_4(paths) 
    elif num == 5:
        gw = GW5.Gridworld_5(paths)
    return gw


### STATE ###

def random_empty_state(state):
    """ return index of random, empty, non/terminal state """
    return random.choice([i for i, s in enumerate(state) if s == 0])

def location_of(key, state_index, grid=False):
    """ returns the first board position of piece """
    idx = list(state_index).index(key)
    return state_to_grid(idx) if grid else idx 


### GRID ###

def state_to_grid(env, state_index):
    """ convert state index to grid coordinate """
    return np.array([state_index // env.width, state_index % env.width])

def grid_to_state(env, grid_coord):
    """ convert grid coordinate into state index """
    return grid_coord[0] * env.width + grid_coord[1]

def is_valid_grid(env, pt):
    """ check if grid coordinate is valid """
    row_check = pt[0] >= 0 and pt[0] < env.height
    col_check = pt[1] >= 0 and pt[1] < env.width
    return row_check and col_check


