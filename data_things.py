import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import paths

from Library import Screen
from Library import DataThings as DT
#import data_classes as DC
from Library import NetworkAPI as NETS

from Library.DataThings import *



### TEXT FILE ###

def load_data():
    """ """
    path = paths.pacman_labels
    images = np.array([np.array(Image.open(os.path.join(path, f))) / 255
              for f in os.listdir(path)])
    #print(images.shape)
    return images


def load_command_info():
    """ load command text info """
    text_data = DT.read_file_split_empty(paths.command_info)
    commands = [DC.Command(lines) for lines in text_data]
    command_dict = dict((cmd.text, cmd) for cmd in commands)
    return command_dict, list(command_dict.keys())



### NETWORK FILE ###

def load_auto(path, auto_name):
    """ """
    return NETS.load_auto(path, auto_name)



### IMAGE FILE ###


def load_images(vocab):
    """ """
    return [Image.open(join(paths.cards_path, v + '.png')) for v in vocab]



