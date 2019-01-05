import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.Learning import ActionUtils


### PARAMS ###

keyDict = dict({'left': (0,-1), 'right': (0,1), 'up': (-1,0), 'down': (1,0),
                'none': (0,0)})
keyDictRev = {v: k for k, v in keyDict.items()}




# default movement
set_1 = ['left', 'right', 'up', 'down']
ms_1 = [GridworldAction(name, keyDict[name]) for name in set_1]


set_2 = ['none', 'left', 'right', 'up', 'down']



ms_1 = ((), (), (), ())




class GridworldAction(ActionUtils.Action):
    """ """

    def __init__(self, name, value):
        """ """
        name = name
        value = value







