import os
import time
import numpy as np
from pynput import keyboard
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

def train_auto_offline():
    """ """
    p = paths.auto_data_path
    ps = [os.path.join(p, file) for file in os.listdir(p)]
    data = DT.load_datas(ps)
    NETS.train_auto(auto_network, data, h, w, n_train=30, kmax_img=1,
                    kmax_cost=1)
    



### LISTENER ###

from pynput import mouse

def mouse_on_click(x, y, button, pressed):
    """ """
    global all_keys, key_listener
    print(button)
    
    

def keyboard_on_release(keyed):
    """ """
    global all_keys
    all_keys += key_to_char(keyed)
    if keyed == Key.enter:
        return False
        

def listen_action():
    """ """
    global all_keys, mouse_lisener
    print('Listening for action...')

    # loop until done
    done = False
    while not done:
        all_keys = ''

        # listen for action
        with keyboard.Listener(on_release=keyboard_on_release) as key_listener:
            try:
                key_listener.join()
            except:
                print('Exception when listening')

        print('Keys: ' + all_keys)
        done = all_keys == 'q'
    print('Ended')
    mouse_listener.stop()

def key_to_char(keyed):
    """ """
    try:
        return keyed.char
    except AttributeError:
        if keyed == Key.space:
            return ' '
        if keyed == Key.enter:
            return ''
        return '_'



#mouse_listener = mouse.Listener(on_click=mouse_on_click)
#mouse_listener.start()
#mouse_listener.stop()
all_keys = ''

    
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
    NETS.new_auto(paths.automation_path, 'test', h, w, [32,16,16,16,16,8,4],
                  batch_size=1)

# train auto network
if 1:
    print('Training AUTO...')
    auto_network = DT.load_auto(base_path, 'AUTO_test_1024_1024_7_256')
    test = Test.Test(test_path, auto_network)
    train_auto_offline()


#ass = listen_action()
#print(ass)

