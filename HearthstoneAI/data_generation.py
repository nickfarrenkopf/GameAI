import os
import time
import numpy as np
from threading import Thread

import paths

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.Computer import Keyboard
from Library.Computer import Screen
from Library.Computer import Mouse
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import _ClassifierAPI as CLASS
from Library.NeuralNetworks.Embedding import _EmbeddingAPI as EMBED



### DATA GENERATION ###

def record_game_data(file_max=100, sleep_time=0.9):
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
        FT.save_image_to_file(env.get_window()[0], filepath, print_me=False)
        # end of loop
        idx += 1
        done = idx > file_max or 'q' in keys
        time.sleep(sleep_time)
    print('Done recording game data')

        
def record_thing(iters, sleep_time=3):
    """ """
    last_data = []
    print('Saving data')
    for i in range(200):
        last_data = record_mana(last_data, iters, i)
        time.sleep(sleep_time)
        if i % 5 == 0:
            print(i)

def record_mana(last_data, iters, i):
    """ """
    ds = np.array([Screen.get_data()])
    # self mana
    dat1 = DT.subdata_xy(ds, 32, 64, 1000, 1262)
    FT.save_image_to_file(dat1[0], 'manaS_{}_{}.png'.format(iters, i))
    # opponent mana
    dat2 = DT.subdata_xy(ds, 32, 64, 67, 1229)
    FT.save_image_to_file(dat2[0], 'manaO_{}_{}.png'.format(iters, i))
    return [dat1, dat2]


### PROGRAM ###

if __name__ == '__main__':


    """ GAME """

    # base folder
    name = 'hearthstone'



    """ GENERATE DATA """
    
    record_thing(0)


