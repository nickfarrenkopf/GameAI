import os
import time
import numpy as np
from PIL import Image
from pynput.keyboard import Key

import paths
import trainer

from Library import Screen
from Library import NetworkAPI as NETS




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
    
def screencap_window(w, name='window', path=paths.images_path):
    """ """
    dats = Screen.get_data()[w[0][0]:w[1][0], w[0][1]:w[1][1], :]
    Screen.save_image(dats, os.path.join(path, '{}.png'.format(name)))
    print('Saved image to images/window.png')
    
def get_window(w):
    """ """
    dats = Screen.get_data()[w[0][0]:w[1][0], w[0][1]:w[1][1], :]
    return dats


def listen():
    """ """
    # params
    count = get_file_count()
    print('Start count: {}'.format(count))
    done = False
    while not done:
        key = Screen.get_key()
        # exit condition
        if key in ['p']:
            done = True
        # action keys
        if key in mapping and key not in dir(key):
            print(mapping[key])
            name = 'listen_{}_{}'.format(count, mapping[key])
            screencap_window(window, name=name, path=paths.labels_path)
            count += 1



def listen_and_train():
    """ """
    print('Listen and Train...')
    size = 512
    done = False
    while not done:
        key = Screen.get_key()
        # exit condition
        if key in ['p']:
            done = True
        # action keys
        if key in mapping:
            ds = np.reshape(get_window(window), (1, size, size, 3))
            NETS.train_auto(auto_network, ds, size, size, kmax_img=0, n_train=3,
                            kmax_cost=2)
            time.sleep(2)
            


def get_file_count():
    """ """
    return len([f for f in os.listdir(paths.labels_path) if 'listen' in f])

def is_special_key(key):
    """ """
    if key not in mapping:
        return False
    return mapping[key]


### PARAMS ###

# key listeners
keys = [Key.up,Key.right,Key.down,Key.left,'z','x','a','s','q','r']
values = ['up','right','down','left','z','x','a','s','q','r']
mapping = dict(zip(keys, values))

window = ((217, 1183), (729, 1695))
#set_window()
#screencap_window(window)

### PROGRAM ###
#from pynput.keyboard import Key

# a b up right down left

#mapp = 

auto_network = trainer.load_auto('AUTO_test_512_512_5_4096')

ds = get_window(window)

#listen_and_train()
