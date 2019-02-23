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
    n = 12800
    h = 512
    w = 512
    le = 3
    h_f = 544
    w_f = 544

    # class data
    class_path = paths.network_path
    size = 128
    n_classes = 4



    """ DATA """

    if 0: # AUTO DATA
        ds = FT.load_images(files[:n])
        #ds = DT.subdata(ds, h, w)
        print('Data shape: {}'.format(ds.shape))

    if 1: # CLASS DATA
        labels = set(('0','1','2'))
        #labels = set(('left','right','up','down'))
        fs, ls = paths.get_filepaths_for_labels(labels, True, True)
        n_classes = len(set(ls))
        print('Data n_files: {}  n_classes: {}\n'.format(len(ls), n_classes))



    """ AUTO """

    if 0: # CREATE
        auto_hidden = [32, 32, 16, 16, 8, 4]
        AUTO.new(paths, name, h, w, auto_hidden, length=le, patch=3, e=1e-8,
                 with_binary=False, reuse_weights=True, print_me=True)

    if 1: # LOAD - NETWORK
        auto_network = AUTO.load(name, paths.load_json())
        #auto_network.print_info()


    if 0: # TRAIN ITER - DATA
        print('Training on data with iters')
        AUTO.train_data_iter(auto_network, ds, h, w, n_train=50, a=1e-4,
                             n_plot=4, plot_r=True, plot_i=True,
                             do_subdata=True, kmax_img=4, kmax_cost=1)
    
    if 0: # TRAIN FULL - DATA
        print('Training on data until finished')
        start = time.time()
        AUTO.train_data_full(auto_network, ds, h, w, alpha=1e-3, n_plot=n//2,
                             do_subdata=True, kmax_cost=20)
        print('Time {}'.format(time.time() - start))
    

    if 0: # TRAIN ITER - PATHS
        print('Training on data with iters')
        AUTO.train_path_iter(auto_network, files, h, w, h_f, w_f, n_train=100,
                             alpha=1e-3, n_plot=n//2, plot_r=True, plot_i=True,
                             kmax_cost=10, kmax_img=20, do_subdata=True)

    if 0: # TRAIN FULL - PATHS
        print('Training on data with iters')
        AUTO.train_path_full(auto_network, files, h, w, h_f, w_f, alpha=1e-4, 
                             n_plot=n//2, do_subdata=True, kmax_cost=20)



    """ EMBEDDING """ 

    if 0: # CREATE
        embed_hidden = [128]
        EMBED.new(paths, name, size, embed_hidden)

    if 1: # LOAD EMBED
        embed_network = EMBED.load(name, paths.load_json())
        embed_network.print_info()

    if 0: # TRAIN
        EMBED.train_path_iter(embed_network, auto_network, files, h, w, a=1e-4,
                              do_subdata=True, n_train=1000, kmax_cost=20)



    """ CLASS """ 

    if 1: # CREATE
        class_hidden = [64, 64]
        CLASS.new(paths, name, size, class_hidden, n_classes)

    if 1: # LOAD CLASS
        class_network = CLASS.load(name, paths.load_json())
        class_network.print_info()

    if 1: # TRAIN
        ls = DT.to_one_hot(ls)
        CLASS.train_path_iter(class_network, auto_network, embed_network,
                              fs, ls, h, w, a=1e-3,
                              batch=128, do_subdata=True, n_train=100,
                              kmax_cost=5)


