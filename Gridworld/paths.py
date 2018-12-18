import os
import json

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import FileThings as FT


### PARAMS ###

# top level
base_path = os.path.dirname(os.path.realpath(__file__))
json_path = os.path.join(base_path, '_json_data.txt')
network_path = os.path.join(base_path, 'network')


### JSON ###

def load_json():
    """ """
    return FT.load_json(json_path)

def write_json(data):
    """ """
    return FT.write_json(json_path, data)

def reset_json():
    """ """
    data = {'network': 
                {'auto': {},
                 'class': {},
                 'reg': {}
                 },
            'learning': {}
            }
    write_json(data)


