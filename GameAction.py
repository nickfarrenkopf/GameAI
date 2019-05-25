import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.Computer import Mouse
from Library.Computer import Keyboard


"""
EMULATOR CONTROLS MAPPING

up, down, left, right
a, b -> z, x
l, r -> a, s
start, select -> q, w

"""


### EMULATOR ###

class EmulatorActions(object):
    """ """

    def __init__(self, action_names):
        """ """
        self.action_names = action_names
        self.action_types = [self.get_action_type() for a in action_names]
        

        self.name_to_type = dict()
        


    def get_action_type(self, action_name):
        """ """
        if action_name in Keyboard.keyDictRev:
            return 'key'
        if action_name in Mouse.mouseDictRev:
            return 'mouse'
        return 'other'



    def take_action(self, name):
        """ """
        if self.name_to_type[name] == 'key':
            Keyboard.press_key(Keyboard.keyDictRev[name])


    def get_names(self):
        """ """
        return [action.name for action in self.actions]

    def get_values(self):
        """ """
        return [action.value for action in self.actions]

    def find_by_name(self, name):
        """ """
        for action in self.actions:
            if action.name == name:
                return action
        print('Name {} not found'.format(name))

    def find_by_value(self, value):
        """ """
        for action in self.actions:
            if action.value == value:
                return action
        print('Value {} not found'.format(value))



### HELPER ###

def load(names):
    """ """
    #return AU.ActionList([EmulatorAction(n, actionDict[n]) for n in names])
    return False


### PARAMS ###

# default action sets
action_sets = dict()
action_sets['set_1'] = load(['left','right','up','down'])
action_sets['set_2'] = load(['left','right','up','down','z','x'])
action_sets['set_3'] = load(['left','right','up','down','z','x','a','s'])
action_sets['mouse_1'] = 0

