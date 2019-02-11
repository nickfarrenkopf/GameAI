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
    size = 256
    n_classes = 4



    """ DATA """

    if 0: # LOAD - DATA
        ds = FT.load_images(files[:n])
        #ds = DT.subdata(ds, h, w)
        print('Data shape: {}'.format(ds.shape))



    """ AUTO """

    if 0: # CREATE
        auto_hidden = [32, 32, 16, 16, 8, 4]
        AUTO.new(paths, name, h, w, auto_hidden, length=le, patch=3, e=1e-8,
                 with_binary=False, reuse_weights=True, print_me=True)

    if 1: # LOAD - NETWORK
        auto_network = AUTO.load(name, paths.load_json())
        auto_network.print_info()


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

    if 0: # LEARN
        print('Learning...?')

    if 0: # TEST
        n_test = 5
        print('Testing auto network for {} random images'.format(n_test))
        for i in range(n_test):
            fp = DT.get_subset(fs, n, True)
            dss = DT.subdata(FT.load_images_4d(fp, h_f, w_f, le), h, w)
            AUTO.check_auto(auto_network, dss, h, w, n // 2, True, True, i, 1)



    """ CLASS """ 

    if 1: # CREATE
        name = 'pacman'
        hidden = [32, 32]
        CLASS.new(paths, name, size, hidden, n_classes)

    if 1: # LOAD DATA
        labels = set(('NA','0','1','2'))
        fs, ls = paths.get_filepaths_for_labels(labels, True)
        #ls = DT.to_one_hot(ls)
        print('N files: {}   N classes: {}'.format(len(ls), len(labels)))

    if 1: # LOAD
        class_network = CLASS.load(name, paths.load_json())
        class_network.print_info()

    if 1: # TRAIN
        ls = DT.to_one_hot(ls)
        CLASS.train_path_iter(class_network, auto_network, fs, ls, h, w, a=1e-3,
                              batch=16, do_subdata=True, n_train=100,
                              kmax_cost=5)







