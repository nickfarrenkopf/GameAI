import os
import numpy as np
from PIL import Image


import paths

from Library import Screen





### WINDOW ###

def set_window(size=512):
    """ """
    global window
    print('Setting main window...')
    x0, y0 = Screen.get_click(' - click top left')
    x1, y1 = Screen.get_click(' - click bottom right')
    x_avg = np.mean([x0, x1], dtype=np.int)
    y_avg = np.mean([y0, y1], dtype=np.int)
    r = size // 2
    window = ((y_avg - r, x_avg - r), (y_avg + r, x_avg + r))
    print(' - set window to {} {}'.format(*window))
    screencap_window(window)
    
def screencap_window(w):
    """ """
    dats = Screen.get_data()[w[0][0]:w[1][0], w[0][1]:w[1][1], :]
    dats = Screen.get_data()[w[0][0]:w[1][0], w[0][1]:w[1][1], :]
    Screen.save_image(dats, os.path.join(paths.images_path, 'window.png'))
    print('Saved image to images/window.png')
    


### PARAMS ###

window = ((217, 1183), (729, 1695))
#set_window()


### PROGRAM ###
from pynput.keyboard import Key

# a b up right down left

#mapp = ['x', 'y', Key.up, Key.right, Key.down, Key.left, 'a', 's',
#        Key.backspace, Key.enter]




