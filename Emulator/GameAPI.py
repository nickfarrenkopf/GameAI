import os
import time
import numpy as np
from threading import Thread

import paths
from learning import EmulatorEnvironment as EE
from learning import EmulatorAgent as EAg
from learning import EmulatorAction as EAc

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.Computer import Mouse
from Library.Computer import Keyboard
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import _ClassifierAPI as CLASS



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
from scipy import spatial

def get_data_thing():
    """ """
    return env.get_window(network=True)

def test_class_network():
    """ """
    fs, labs = paths.get_filepaths_for_labels(set(('0','1','2')), False, False)
    ld = FT.load_images(fs)
    ld = auto_network.get_latent(ld)
    #labs = DT.to_one_hot(ls)
    for i in range(200):
        deed = auto_network.get_latent(env.get_window(network=True))
        preds = most_similar(deed[0], ld, labs)
        #preds = class_network.get_preds(deed)
        print(preds)
        time.sleep(0.2)


def get_similar(vec, vectors, top_n=5):
    """ """
    sims = [(i,1-spatial.distance.cosine(v, vec)) for i,v in enumerate(vectors)]
    sims.sort(key=lambda x: x[1])
    returns = sims[-top_n:]
    returns = list(reversed(returns))
    return sims[-top_n:]

def most_similar(data, labeled_data, labels):
    """ """
    sims = get_similar(data, labeled_data, top_n=5)
    idxs = [s[0] for s in sims]
    labels = [labels[i] for i in idxs]
    return labels
    


### PROGRAM ###

if __name__ == '__main__':


    """ GAME """

    # base folder
    name = 'pacman'
    paths.set_base(name)
    files = FT.get_filepaths(paths.image_path)


    """ DATA """

    if 0: # AUTO DATA
        ds = FT.load_images(files[:5000])
        print('Data shape: {} {} {}'.format(ds.shape, ds.max(), ds.min()))


    """ LOAD NETWORKS """

    if 1:
        auto_network = AUTO.load(name, paths.load_json())
        print('{} auto network loaded'.format(name))

    if 1:
        class_network = CLASS.load(name, paths.load_json())


    """ OTHER """

    if 1: # test image
        env = EE.Emulator(paths, 'pacman')
        env.save_window()
        #dd = env.get_window(network=True)
    

    if 1:
        key = 0
        actions = EAc.action_sets['emulator_4']
        ag = EAg.EmulatorAgent(env, key, actions)
        ag.play()



    #AUTO.watch_auto(auto_network, get_data_thing)
    
    #test_class_network()


