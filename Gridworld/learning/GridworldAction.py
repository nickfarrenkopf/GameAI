import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.Learning import ActionUtils as AU


### GRIDWORLD ###

class GridworldAction(AU.Action):
    """ """

    def __init__(self, name, value):
        """ """
        AU.Action.__init__(self, name, value)
        self.x = value[0]
        self.y = value[1]

    # def get_names
    # def get_values
    # def find_by_name
    # def find_by_value


### HELPER ###

def load(names):
    """ """
    return AU.ActionList([GridworldAction(n, actionDict[n]) for n in names])


### PARAMS ###

# action dictionarites
actionDict = dict({'left': (0,-1), 'right': (0,1), 'up': (-1,0), 'down': (1,0),
                   'none': (0,0)})
actionDictRev = {v: k for k, v in actionDict.items()}

# default movement
actionset_1 = load(['left', 'right', 'up', 'down'])
actionset_2 = load(['none', 'left', 'right', 'up', 'down'])


