import os
from os.path import join

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import FileThings as FT


### PARAMS ###

# top level
base_path = os.path.dirname(os.path.realpath(__file__))

# game specific
game_path = ''
json_path = ''
image_path = ''
network_path = ''
labels_path = ''


### PATHS ###

def set_base(game):
    """ """
    global game_path, json_path, image_path, network_path, labels_path
    game_path = join(base_path, 'data', game)
    json_path = join(game_path, 'json_data.txt')
    image_path = join(game_path, 'image')
    network_path = join(game_path, 'network')
    labels_path = join(game_path, 'custom_labels.txt')


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


