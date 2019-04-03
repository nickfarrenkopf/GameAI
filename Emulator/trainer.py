import os
import time
import numpy as np

import paths

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import _ClassifierAPI as CLASS
from Library.NeuralNetworks.Embedding import _EmbeddingAPI as EMBED


### PROGRAM ###

if __name__ == '__main__':


    """ GAME """

    # base folder
    name = 'pacman'
    paths.set_base(name)
    files = FT.get_filepaths(paths.image_path)

    # data params
    n = 128
    h = 512
    w = 512
    le = 3
    h_f = 544
    w_f = 544

    # class data
    class_path = paths.network_path
    size = 64
    n_classes = 4



    """ DATA """

    if 0: # AUTO DATA
        ds = FT.load_images(files[:n])
        ds = DT.subdata(ds, h, w)
        print('Data shape: {} {} {}'.format(ds.shape, ds.max(), ds.min()))

    if 0: # CLASS DATA
        labels = set(('0','1','2'))
        #labels = set(('left','right','up','down'))
        fs, ls = paths.get_filepaths_for_labels(labels, True, True)
        n_classes = len(set(ls))
        print('Data n_files: {}  n_classes: {}\n'.format(len(ls), n_classes))



    """ AUTO """

    if 1: # CREATE
        #auto_hidden = [32, 32, 32, 16, 8, 4]
        paths.reset_json()
        hidde_encode = [2, 3, 4, 5, 6, 7]
        pools_encode = [2, 2, 2, 2, 2, 2]
        hidde_decode = [6, 5, 4, 3, 2, 1]
        pools_decode = [1, 1, 1, 1, 1, 1]
        hidden_dense = [512, 512]
        n_latent = 64
        hidden_encode = [2 ** i for i in hidde_encode]
        hidden_decode = [2 ** i for i in hidde_decode]
        AUTO.new(paths, name, h, w, length=le, patch=5, print_me=True,
                 hidden_encode=hidden_encode, pools_encode=pools_encode,
                 hidden_latent=hidden_dense, n_latent=n_latent,
                 pools_decode=pools_decode, hidden_decode=hidden_decode)

    if 1: # LOAD - NETWORK
        auto_network = AUTO.load(name, paths.load_json())
        #auto_network.print_info()


    if 0: # TRAIN ITER - DATA
        print('Training on data with iters')
        AUTO.train_data_iter(auto_network, ds, h, w, n_train=10000, a=1e-3,
                             n_plot=20, plot_r=True, plot_i=True,
                             do_subdata=False, kmax_img=2000, kmax_cost=50)

    if 1: # TRAIN ITER - PATHS
        print('Training on data with iters')
        AUTO.train_path_iter(auto_network, files, h, w, n_train=100000,
                             a=1e-3, n_plot=16, plot_r=True, plot_i=True,
                             kmax_cost=500, kmax_img=2000, do_subdata=True)


    """ EMBEDDING """ 

    if 0: # CREATE
        embed_hidden = [128]
        EMBED.new(paths, name, size, embed_hidden)

    if 0: # LOAD EMBED
        embed_network = EMBED.load(name, paths.load_json())
        embed_network.print_info()

    if 0: # TRAIN
        EMBED.train_path_iter(embed_network, auto_network, files, h, w, a=1e-4,
                              do_subdata=True, n_train=1000, kmax_cost=20)



    """ CLASS """ 

    if 0: # CREATE
        class_hidden = [64, 64]
        CLASS.new(paths, name, size, class_hidden, n_classes)

    if 0: # LOAD CLASS
        class_network = CLASS.load(name, paths.load_json())
        class_network.print_info()

    if 0: # TRAIN
        ls = DT.to_one_hot(ls)
        CLASS.train_path_iter(class_network, auto_network, embed_network,
                              fs, ls, h, w, a=1e-3,
                              batch=128, do_subdata=True, n_train=100,
                              kmax_cost=5)


