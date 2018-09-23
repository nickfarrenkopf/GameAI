import os
import time
import numpy as np
from pynput.keyboard import Key

import paths

from Library.General import Screen
from Library.General import DataThings as DT
from Library.Network import NetworkAPI as NETS
from Library.TA import When
from Library.TA import What
from Library.TA import Where
from Library.TA import Test


### API ###

def run_program(train_me=False, save_me=True):
    """ """
    
    # loop until done
    done = False
    while not done:

        # RL EVENTS
        done = True


def listen_for_actions():
    """ """
    done = False
    while not done:
        


def listen_action():
    """ """
    pass
    # keyboard keyS and mouse action listener



### TRAIN ####

def train_networks():
    """ """
    print('Training')

def train_auto_network_online():
    """ """
    done = False
    count = 0
    while not done:
        data = np.reshape(test.get_window(), (-1, h, w, 3))
        print(data.shape)
        NETS.train_auto(auto_network, data, h, w, n_train=5, kmax_img=100,
                        kmax_cost=5)
        time.sleep(0.5)
        count += 1
        if count % 3 == 0:
            NETS.plot_middle(auto_network, data, h, w, n_plot=1, count=count)



### HELPER ###


def record_game_data():
    """ """
    # starting file counter
    count = 0
    if len(os.listdir(game.state_path)) > 0:
        count = int(os.listdir(game.state_path)[-1].split('_')[1]) + 1
    print('Start count: {}'.format(count))
    # wait for key
    done = False
    while not done:
        key = Screen.get_key()
        print(key)
        # if in whitelist, take screencap
        if key in game.action_keys or key in game.label_keys:
            text = '0' * (4 - len(str(count))) + str(count)
            names = ['image_{}_{}_0'.format(text, values[keys.index(key)])]
            windows = [game.get_window()]
            if key in game.action_keys:         
                for i in range(1, 3):
                    text = '0' * (4 - len(str(count))) + str(count)
                    names.append('image_{}_{}_{}'.format(text,
                                                         values[keys.index(key)],
                                                         i))
                    windows.append(game.get_window())
                    time.sleep(0.1)
            for name, window in zip(names, windows):
                path = os.path.join(game.state_path, name + '.png')
                Screen.save_image(window, path)
            count += 1
        done = key == 'p'



from threading import Timer
from pynput.keyboard import Key
exit_loop = True

def end_of_time():
    """ """
    exit_loop = True

def get_keys():
    """ """
    done = False
    keys = []
    while not done:
        t = Timer(1, end_of_time)
        t.start()
        key = Screen.get_key()
        keys.append(key)
        done = key == Key.enter
    want_keys = []
    for key in keys:
        if type(key) == str:
            want_keys.append(key)
        if key == Key.space:
            want_keys.append(' ')
    print(''.join(want_keys))
    return want_keys
    
    


### PARAMS ###

# location
base_path = paths.automation_path
test_path = os.path.join(base_path, 'test_1')

# screen size
h = 1024
w = 1024


### PROGRAM ###

# load test
if 0:
    auto_network = DT.load_auto(base_path, 'AUTO_test_1024_1024_7_256')
    test = Test.Test(test_path, auto_network)


# create auto network
if 0:
    print('Creating AUTO...')
    NETS.new_auto(paths.automation_path, 'test', h, w, [32,16,16,8,8,8,4],
                  batch_size=1)

# train auto network
if 0:
    print('Training AUTO...')
    auto_network = DT.load_auto(base_path, 'AUTO_test_1024_1024_7_256')
    test = Test.Test(test_path, auto_network)
    train_auto_network_online()


get_keys()
    
