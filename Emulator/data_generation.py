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
from Library.Computer import Keyboard
from Library.Computer import Screen
from Library.Computer import Mouse
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import _ClassifierAPI as CLASS
from Library.NeuralNetworks.Embedding import _EmbeddingAPI as EMBED


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



def get_data_thing():
    """ """
    return env.get_window(network=True)


### PROGRAM ###

if __name__ == '__main__':


    """ GAME """

    # base folder
    name = 'pacman'
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
    record_thing(0)


        
    

    if 0:
        auto_network = AUTO.load(name, paths.load_json())

    if 0:
        class_network = CLASS.load(name, paths.load_json())

    if 0:
        embed_network = EMBED.load(name, paths.load_json())

    if 0:
        env = EE.Emulator(paths, name)


    """ OTHER """

    #fs, ls = paths.get_filepaths_for_labels(set(('1','2','0')))
    #lss = DT.to_one_hot(ls)

    #AUTO.TEST.plot_middle_runtime(auto_network, env.get_window)
    
    if 0:
        CLASS.TEST.print_class_runtime(class_network, auto_network,
                                       embed_network, env.get_window)

    if 0:
        EMBED.TEST.plot_middle_runtime(auto_network, embed_network,
                                       env.get_window)
    
    #record_game_data(file_max=3000, sleep_time=0.1)


    #env.save_state_image()
