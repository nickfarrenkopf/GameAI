import os
import time
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread

import paths
from learning import EmulatorEnvironment as EE

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.Computer import Screen
from Library.Computer import Keyboard
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import _ClassifierAPI as CLASS





class DataPoint(object):
    """ """

    def __init__(self):
        """ """
        self.name = ''
        self.episode = ''
        self.idx = ''
        self.keys = ''


### DATA GENERATION ###

def record_game_data(file_max=100, sleep_time=0.9):
    """ """
    # start keyboard listener
    t = Thread(target=Keyboard.start_constant_listener, args=())
    t.start()
    # get initial conditions
    filenames = os.listdir(paths.image_path)
    episodes = sorted(list(set([f.split('_')[1] for f in filenames])))
    episode = int(episodes[-1]) + 1 if len(episodes) > 0 else 0
    idx = 0
    print('Episode count: {}'.format(episode))
    # loop until done
    done = False
    while not done:
        # get currently pressed keys
        keys = ','.join(Keyboard.get_pressed())
        keys = keys if len(keys) > 0 else 'NA'
        # save to file
        filename = '{}_{}_{}_{}.png'.format(name, episode, idx, keys)
        filepath = os.path.join(paths.image_path, filename)
        FT.save_image_to_file(env.get_window(), filepath, print_me=False)
        # end of loop
        idx += 1
        done = idx > file_max or 'q' in keys
        time.sleep(sleep_time)
    print('Done recording game data')




def prepend_zeros():
    """ """
    pass


def get_data_thing():
    """ """
    #return np.random.random((4, 512, 512, 3))
    wind = np.reshape(env.get_window(), (-1, h, w, le))
    #print(wind.shape)
    #mid = np.reshape(auto_network.get_flat(wind), auto_network.plot_shape)
    return wind


### PROGRAM ###

if __name__ == '__main__':


    """ GAME """

    # base folder
    name = 'test'
    paths.set_base(name)
    files = FT.get_filepaths(paths.image_path)

    # auto data
    n = 128
    h = 512
    w = 512
    le = 3
    h_f = 544
    w_f = 544
    

    """ LOAD NETWORKS """

    if 1:
        auto_network = AUTO.load(name, paths.load_json())

    if 0:
        class_network = CLASS.load(name, json_data)


    """ OTHER """



    #t = Thread(target=Keyboard.start_constant_listener, args=())
    #t.start()



    env = EE.Emulator(paths, 'test')
    #env.save_state_image()

    


    #key = Keyboard.get_key()

    #for i in range(10):
    #    print(Keyboard.get_pressed())
    #    time.sleep(0.1)



    AUTO.watch_auto(auto_network, get_data_thing)
    


    #record_game_data()



