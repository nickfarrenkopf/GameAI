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
from Library.Computer import Screen
from Library.Computer import Keyboard
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import _ClassifierAPI as CLASS


### DATA GENERATION ###

def record_game_data(file_max=100, sleep_time=0.5):
    """ """
    # start keyboard listener
    t = Thread(target=Keyboard.get_key_multiple, args=())
    t.start()
    # get initial conditions
    filenames = os.listdir(paths.image_path)
    episodes = sorted(list(set([f.split('_')[1] for f in filenames])))
    episode = int(episodes[-1]) + 1 if len(episodes) > 0 else 0
    idx = 0
    # loop until done
    done = False
    while not done:
        # get currently pressed keys
        keys = ','.join(Keyboard.get_currently_pressed())
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
    prepend_zeros_idx()

def prepend_zeros_idx():
    """ """
    count = 0
    filenames = os.listdir(paths.image_path)
    episodes = [int(f.split('_')[1]) for f in filenames]
    for i in range(len(set(episodes))):
        files = [filenames[j] for j, e in enumerate(episodes) if e == i]
        max_len = len(str(len(files)))
        print('Episode: {}   n_files: {}'.format(i, len(files)))
        for file in files:
            params = file.split('_')
            middle = '0' * (max_len - len(params[2])) + params[2]
            str_name = '_'.join(params[:2] + [middle] + params[3:])
            if str_name != file:
                os.rename(os.path.join(paths.image_path, file),
                          os.path.join(paths.image_path, str_name))
                count += 1
    print('Files renamed {}'.format(count))


### PROGRAM ###

if __name__ == '__main__':

    """ GAME """
    name = 'pacman'
    paths.set_base(name)
    h, w, le = (160, 160, 3)

    """ RL """
    if 1:
        env = EE.Emulator(paths, name)

    record_game_data(file_max=10000)


