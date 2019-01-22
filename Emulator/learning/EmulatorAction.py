import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Keyboard
from Library.Learning import ActionUtils as AU


### EMULATOR ###

class EmulatorAction(AU.Action):
    """ """

    def __init__(self, name, value):
        """ """
        AU.Action.__init__(self, name, value)
        self.x = value[0]
        self.y = value[1]

    def take_action(self, action_name):
        """ """
        Keyboard.press_key(actionDict[action_name])


### HELPER ###

def load(names):
    """ """
    return AU.ActionList([EmulatorAction(n, actionDict[n]) for n in names])


### PARAMS ###

# action dictionarites
actionDict = keyDictRev
actionDictRev = {v: k for k, v in actionDict.items()}

# default movement
actionset_1 = load(['left', 'right', 'up', 'down'])
actionset_2 = load(['left', 'right', 'up', 'down', 'z', 'x'])


