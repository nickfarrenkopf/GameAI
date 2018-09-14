import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import paths

from Library.General import Screen
from Library.General import DataThings as DT

from Library import NetworkAPI as NETS


from Library.General.DataThings import *


from os.path import join


a = 395
b = 286
c = 3

def load_hs_data():
    """ """
    width = height = 620
    vocab = os.listdir(paths.hs_data_path)
    imgs = [Image.open(join(paths.hs_data_path, v)) for v in vocab]
    imgs = [img.resize((width, height), Image.ANTIALIAS) for img in imgs]
    data = np.array([np.array(img)[:, :, :3] / 255 for img in imgs])
    return data


### TEXT FILE ###

def load_data():
    """ """
    path = paths.pacman_gamedata_path
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



