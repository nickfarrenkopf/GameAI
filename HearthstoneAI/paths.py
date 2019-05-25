import os
from os.path import join

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import FileThings as FT


# top level
base_path = os.path.dirname(os.path.realpath(__file__))
json_path = join(base_path, '_json_data.txt')
data_path = join(base_path, 'data')

# data types
images_path = join(data_path, 'images')
labels_path = join(data_path, 'labels')
network_path = join(data_path, 'networks')



### HELPER ###

def get_image_path(name):
    """ """
    return join(images_path, name)

def get_label_path(name):
    """ """
    return join(labels_path, '{}_labels.txt'.format(name))

def get_labels(name):
    """ """
    return FT.read_file_csv(get_label_path(name))


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


