import paths
import GW1
import GW2
import GW3
import GW4
import GW5

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.NeuralNetworks.Regression import OptimizerRegression as REG


### GRIDWORLD ###

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

def learn_gridworld_params(num):
    """ """
    gridworld = get_gridworld(num)
    gridworld.set_training_params(run_train, run_pred, train_start, with_decay)
    gridworld.initialize()
    gridworld.load_value_data()



### PARAMS ###

# gridworld
run_train = 1
run_pred = 0
train_start = 0
with_decay = 1







#done = False
#while not done:
#    gridworld.iterate()
#    done = True



