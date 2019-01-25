import os
import time
import numpy as np
from threading import Thread

import paths
from learning import EmulatorEnvironment as EE

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.General import Screen
from Library.Computer import Keyboard
from Library.NeuralNetworks.Autoencoder import AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import ClassifierAPI as CLASS


### PROGRAM ###



# data point for images
class DataPoint(object):
    """ """

    def __init__(self):
        """ """
        self.name = ''
        self.episode = ''
        self.idx = ''
        self.keys = ''



def prepend_zeros():
    """ """
    pass


def rgd():
    """ """
    files = 0
    filenames = files
    
    episodes = sorted(list(set([f.split('_')[0] for f in filenames])))
    episode = episodes[-1] + 1 if len(episodes) > 0 else 0
    print(episodes)
    print(episode)
    
    idx = 0
    keys = []

    done = False
    while not done:
        print('Checking things')

        # image names params
        idx += 1
        keys = []
        filename = '{}_{}_{}_{}'.format(name, episode, idx, keys)
        filepath = os.path.join(paths.image_path, filename)

        # save image



def record_game_data(path):
    """ """
    # starting file counter
    count = 0
    if len(os.listdir(path)) > 0:
        count = int(os.listdir(path)[-1].split('_')[1]) + 1
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

if __name__ == '__main__':


    """ GAME """

    # base folder
    name = 'test'
    paths.set_base(name)
    files = FT.get_filepaths(paths.image_path)

    # auto data
    n = 16
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



    #env = EE.Emulator(paths, 'test')
    #env.save_state_image()

    


    #key = Keyboard.get_key()

    #for i in range(10):
    #    print(Keyboard.get_pressed())
    #    time.sleep(0.1)




