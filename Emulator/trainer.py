import os
import numpy as np
from random import shuffle

import paths

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.NeuralNetworks.Autoencoder import AutoencoderAPI as AUTO


### PROGRAM ###

if __name__ == '__main__':


    """ GAME """

    name = 'hearthstone'
    paths.set_base(name)
    json_data = paths.load_json()

    auto_path = paths.network_path
    n = 16
    h = 1024
    w = 1024
    le = 3
    h_d = 1024
    w_d = 1024


    """ AUTO """

    if 0: # LOAD
        auto_network = AUTO.load_auto(name, json_data)

    if 0: # CREATE
        hidden = [20, 16, 16, 16, 16, 16, 16, 16]
        json_data = AUTO.new_auto(paths, name, h, w, hidden, length=3,
                                  with_binary=True)

    if 0: # TRAIN - DATA
        auto_network = AUTO.load_auto(name, json_data)
        ds = FT.load_images_4d(paths.get_game_images()[7:7+n], h_d, w_d, le)
        print('Data shape: {}'.format(ds.shape))
        AUTO.train_auto_data(auto_network, ds, h, w, n_train=10000,
                             kmax_img=25, kmax_cost=10, alpha=0.001)

    if 1: # TRAIN - PATHS
        data_path = FT.get_filepaths(paths.small_ds) #paths.get_game_images()
        auto_network = AUTO.load_auto(name, json_data)
        AUTO.train_auto_paths(auto_network, data_path,
                              h, w, n_train=10000, alpha=0.001, n_plot=n,
                              kmax_img=2, kmax_cost=2)

    if 0: # LEARN
        ds = np.reshape(DT.load_datas(paths.get_game_images()[:16]), (-1, h2, w2, 3))
        print('Data shape: {}'.format(ds.shape))
        AUTO.learn_auto_data(auto_network, ds, h, w, kmax_cost=5, slope_min=1e-3,
                             slope_count_max=10)




