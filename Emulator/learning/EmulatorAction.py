import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.Computer import Mouse
from Library.Computer import Keyboard


### API ###

def take_actions(actions):
    """ """
    for data in actions:
        if type(data) == str:
            take_action(data)
        else:
            take_action(data[0], data[1:])

def take_action(name, extra=None):
    """ """
    # no action
    if name == 'NA':
        return
    # mouse actions
    if name in Mouse.mouseDictRev:
        take_mouse_action(name, extra)
    # keyboard actions
    elif name in Keyboard.keyDictRev:
        take_keyboard_actions(name)
    # invalid action
    else:
        print('No action taken for {} {}'.format(name, extra))


### HELPER ###

def take_mouse_actions(name, extra):
    """ """
    # left click and drag
    if 'click_drag' in name: 
        Mouse.click_drag(*extra)
    # left/right/middle click
    elif 'click' in name:
        Mouse.click(extra[0], extra[1], name)

def take_keyboard_actions(name):
    """ """
    # multiple keys
    if ',' in name:
        Keyboard.press_keys(name.split(','))
    # single key
    else:
        Keyboard.press_key(name)


### PARAMS ###


"""
EMULATOR CONTROLS MAPPING
   a, b -> z, x
   l, r -> a, s
   start, select -> q, w

"""

# common keyboard set combos
key_combos_0 = ['left', 'right', 'up', 'down']
key_combos_1 = ['left,up', 'left,down', 'right,up', 'right,down']

# default action sets
sets = dict()
sets['emulator_1'] = ['NA'] + key_combos_0
sets['emulator_2'] = ['NA'] + key_combos_0 + key_combos_1
sets['mouse_1'] = ['NA'] + ['left_click']
sets['mouse_2'] = ['NA'] + ['left_click','left_click_drag']


