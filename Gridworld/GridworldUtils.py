import paths
from learning.environments import GW1
from learning.environments import GW2
from learning.environments import GW3
from learning.environments import GW4
from learning.environments import GW5


### HELPER ###

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
    

