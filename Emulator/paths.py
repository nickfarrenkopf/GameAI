import os
import json
from os.path import join
from random import shuffle
from itertools import chain

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import FileThings as FT


### PARAMS ###

# top level
base_path = os.path.dirname(os.path.realpath(__file__))
json_path = join(base_path, '_json_data.txt')

image_path = ''
network_path = ''


### PATHS ###

def set_base(game):
    """ """
    global image_path, network_path
    image_path = join(base_path, 'data', game, 'image')
    network_path = join(base_path, 'data', game, 'network')

def get_filepaths_for_labels(wanted_labels, shuffle_me=True):
    """ """
    filepaths, labels = [], []
    filenames = os.listdir(image_path)
    label_set = set(chain.from_iterable([get_labels(f) for f in filenames]))
    for f in filenames:
        ls = get_labels(f)
        if len(list(label_set & set(ls))) > 0 and ls[0] in wanted_labels:
            filepaths.append(os.path.join(image_path, f))
            labels.append(ls[0])
    if shuffle_me:
        idxs = list(range(len(filepaths)))
        shuffle(idxs)
        filepaths = [filepaths[i] for i in idxs]
        labels = [labels[i] for i in idxs]
    return filepaths, labels

def get_labels(file):
    """ """
    return file.split('.')[0].split('_')[-1].split(',')



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


