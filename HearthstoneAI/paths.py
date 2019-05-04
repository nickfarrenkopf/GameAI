import os
from os.path import join

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import FileThings as FT


# top level
base_path = os.path.dirname(os.path.realpath(__file__))
json_path = join(base_path, '_json_data.txt')
params_path = join(base_path, '_hearthstone_params.txt')
data_path = join(base_path, 'data')

# data types
network_path = join(data_path, 'networks')
images_path = join(data_path, 'images')

# text data


# images
images_mana = join(images_path, 'mana')
images_hero = join(images_path, 'hero')
#cards_path = join(data_path, 'cards')



# text
#card_info = join(data_path, 'card_info.txt')
#command_info = join(data_path, 'command_info.txt')
#node_info = join(data_path, 'node_info.txt')




### JSON ###

def load_params():
    """ """
    return FT.load_json(params_path_json)



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
