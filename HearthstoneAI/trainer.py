import os
import time
import numpy as np

import paths
import ScreenWatcher as SW

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import _ClassifierAPI as CLASS
from Library.NeuralNetworks.Embedding import _EmbeddingAPI as EMBED


### HELPER ###

def plot_things():
    """ """
    outs = auto_network.get_mu(ds)
    outs = outs - np.min(outs, axis=0)
    outs = outs / np.max(outs, axis=0)
    labs = ['{}_{}'.format(*o) for o in outs]
    DT.plot_data_multiple(outs, labels=labs, n_x=6, n_y=6)
    

### PROGRAM ###

if __name__ == '__main__':


    """ GAME """

    # base folder
    name = 'heo'
    files = FT.get_filepaths(paths.images_hero)

    n = 1280


    """ DATA """

    if 1: # AUTO DATA
        ds = FT.load_images(files[:n])
        print('Data shape: {} {} {}'.format(ds.shape, ds.max(), ds.min()))


    """ AUTO """

    if 0: # CREATE
        paths.reset_json()
        hidden_encode = [4, 4, 4, 5, 5, 5]
        pools_encode = [2, 1, 2, 1, 2, 1]
        hidden_decode = [5, 5, 4, 3]
        pools_decode = [1, 1, 1, 1]
        hidden_dense = [256, 256]
        hidden_encode = [2 ** i for i in hidden_encode]
        hidden_decode = [2 ** i for i in hidden_decode]
        n_latent = 4
        AUTO.new(paths, name, 16, 32, length=3, patch=4, print_me=True,
                 hidden_encode=hidden_encode, pools_encode=pools_encode,
                 hidden_latent=hidden_dense, n_latent=n_latent,
                 pools_decode=pools_decode, hidden_decode=hidden_decode)

    if 1: # LOAD - NETWORK
        #auto_network = AUTO.load(name, paths.load_json())
        ws = SW.load_watchers(i=3,j=4,nets=True)[0]
        ws.create_network()
        auto_network = ws.auto_network
        auto_network.print_info()

    if 1: # TRAIN ITER - DATA
        print('Training on data with iters')
        AUTO.train_data_iter(auto_network, ds, 16, 32, n_train=50000, a=1e-3,
                             n_plot=20, plot_r=True, plot_i=True,
                             do_subdata=False, kmax_img=1000, kmax_cost=100)

    

    if 0: # TRAIN ITER - PATHS
        print('Training on data with paths')
        AUTO.train_path_iter(auto_network, files, h, w, n_train=100000,
                             a=1e-3, n_plot=16, plot_r=True, plot_i=True,
                             kmax_cost=100, kmax_img=200, do_subdata=False)




    ### SCREEN WATCHER ###

    if 0: # LOAD WATCHERS
        watchers = SW.load_watchers()



