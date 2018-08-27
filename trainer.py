import os
import time
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

import paths
import data_things as dt

from Library import Screen
from Library import NetworkAPI as NETS


### FILE ###

def load_auto(auto_name):
    """ """
    return NETS.load_auto(paths.network_path, auto_name)



### TRAIN ###

### TEST CLICK ###

def auto_click(auto_network, size=128, n_times=5):
    """ """
    # take n clicks
    datas = []
    for _ in range(n_times):
        x, y = Screen.get_click()
        if x < 0 or y < 0:
            return
        shape = (1, size, size, 3)
        datas.append(np.reshape(Screen.subimage(Screen.get_data(), x, y,
                                                size, size), shape))
    #save_path = os.path.join(paths.images, 'auto_test_click.png')
    datas = np.reshape(datas, (n_times, size, size, 3))
    print(datas.shape)
    NETS.plot_before_after(auto_network, datas, size, 'test_click')



### HELPER ###

def get_clicks(n_click=10):
    """ """
    points = []
    for i in range(n_click):
        print('{}'.format((n_click - i)))
        points.append(Screen.get_click())
    return points


def whats(word, n_pos=5, n_neg=10):
    """ """
    net = load_binary(auto_network, word)
    train_node(net, n_pos=n_pos, n_neg=n_neg)
    net.save_network()




    


def train_auto_full():
    """ """
    network = NETS.load_auto(paths.auto_path, 'AUTO_look_320_480_5_2400')
    for k in range(n_train):
        screen_data = []
        for j in range(n_plot):
            screen_data.append(Screen.get_fullscreen())
            time.sleep(0.2)
        subdata = np.array(screen_data)
        NETS.train_auto(network, subdata, 320, 480, n_train=100, alpha=0.0001,
                        kmax_cost=10, kmax_img=0)
        NETS.plot_before_after(network, subdata, 320, 480, count=k)


    

### PROGRAM ###

auto_network = 'hello'
#x

# -- NEW AUTO --
## NETS.new_auto(paths.auto_path, 'test', 512, 512, [64, 32, 32, 32, 16, 16])
## NETS.new_auto(paths.auto_path, 'test', 512, 512, [32, 16, 8, 4])
## NETS.new_auto(paths.auto_path, 'look', 128, 128, [16, 16])

# -- LOAD AUTO --
#auto_network = NETS.load_auto(paths.network_path, '')
#auto_network = NETS.load_auto(paths.auto_path, 'AUTO_test_512_512_5_4096')
#auto_network = NETS.load_auto(paths.auto_path, 'AUTO_what_128_4_256')
#auto_network = NETS.load_auto(paths.auto_path, 'AUTO_test_1_128_128_4_256')
#auto_network = NETS.load_auto(paths.auto_path, 'AUTO_look_320_480_5_2400') 


#if True:
#    auto_network = load_auto('AUTO_test_512_512_5_4096')
##    ds = dt.load_data()
#   NETS.train_auto(auto_network, ds, 512, 512, kmax_img=5, n_train=100)

# -- TRAIN AUTO --
#NETS.train_auto(auto_network, ds, 128, kmax_img=100)
#NETS.train_auto(auto_network, ds, 128, 128, kmax_img=50, n_train=1000)
#NETS.train_auto_full(auto_network)

# -- TEST AUTO --
#NETS.plot_before_after(auto_network, ds, 128, 128)
#NETS.plot_middle(auto_network, ds, 128, 128)



from PIL import Image
import Library.DataThings as DTS






