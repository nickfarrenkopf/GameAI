import os
import json
from os.path import join


import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import FileThings as FT

### PARAMS ###

# top level
base_path = os.path.dirname(os.path.realpath(__file__))
json_path = join(base_path, '_json_data.txt')
image_path = ''
network_path = ''
small_ds = join(base_path, 'old', 'small_ds')


### SET PATHS ###

def set_base(game):
    """ """
    global image_path, network_path
    image_path = join(base_path, 'data', game, 'image')
    network_path = join(base_path, 'data', game, 'network')
    

### GET PATHS ###

def get_game_images():
    """ """
    files = [join(image_path, f) for f in os.listdir(image_path)]
    return files


### JSON ###

def load_json():
    """ """
    return FT.load_json(json_path)

def write_json(data):
    """ """
    return FT.write_json(json_path, data)

def reset_json():
    """ """
    data = FT.base_json()
    write_json(data)


