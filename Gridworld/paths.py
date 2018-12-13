import os
import json


### PARAMS ###

# top level
base_path = os.path.dirname(os.path.realpath(__file__))
json_path = os.path.join(base_path, '_json_data.txt')
network_path = os.path.join(base_path, 'network')


### JSON ###

.def load_json():
    """ """
    with open(json_path, 'r') as file:  
        return json.load(file)

def write_json(data):
    """ """
    with open(json_path, 'w') as file:  
        json.dump(data, file, indent=3)

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


