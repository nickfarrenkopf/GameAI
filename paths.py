import os
import json
from os.path import join


### PARAMS ###

# top level
base_path = os.path.dirname(os.path.realpath(__file__))
json_path = join(base_path, 'data', 'json_data.txt')
image_path = ''
network_path = ''



### SET PATHS ###

def set_base(game):
    """ """
    global image_path, network_path
    image_path = join(base_path, 'data', 'image', game)
    network_path = join(base_path, 'data', 'network', game)
    

### GET PATHS ###

def get_game_images():
    """ """
    files = [join(image_path, f) for f in os.listdir(image_path)]
    return files

def get_audio_image_files():
    """ """
    global image_audio_path
    files = [join(image_audio_path, f) for f in os.listdir(image_audio_path)
             if '.png' in f]
    return files


### JSON ###

def load_json():
    """ """
    global json_path
    if not os.path.exists(json_path):
        print('JSON DNE {}'.format(json_path))
        return {}
    with open(json_path, 'r') as file:  
        return json.load(file)

def write_json(data):
    """ """
    global json_path
    with open(json_path, 'w') as file:  
        json.dump(data, file, indent=3)

def reset_json():
    """ """
    data = {'network': 
                {'auto': {},
                 'class': {},
                 'reg': {}
                 }
            }
    write_json(data)


