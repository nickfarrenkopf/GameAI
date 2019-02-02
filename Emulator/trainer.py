import os
import time
import numpy as np

import paths

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.NeuralNetworks import TrainUtils as TU
from Library.NeuralNetworks.Autoencoder import _AutoencoderAPI as AUTO
from Library.NeuralNetworks.Classifier import _ClassifierAPI as CLASS


### PROGRAM ###

if __name__ == '__main__':


    """ GAME """

    # base folder
    name = 'test'
    paths.set_base(name)
    files = FT.get_filepaths(paths.image_path)

    # auto data
    n = 10000
    h = 512
    w = 512
    le = 3
    h_f = 544
    w_f = 544



    """ AUTO """

    if 1: # CREATE
        auto_hidden = [32, 32, 32, 32, 32, 32, 32, 64]
        #auto_hidden = [16, 16, 16, 16, 16, 16, 16, 64]
        #auto_hidden = [64, 64, 64, 64]
        AUTO.new(paths, name, h, w, auto_hidden, length=le, patch=3, e=1e-8,
                 with_binary=False, reuse_weights=False, print_me=True,
                 hidden_feedforward=[])

    if 1: # LOAD - NETWORK
        auto_network = AUTO.load(name, paths.load_json())

    if 1: # LOAD - DATA
        ds = FT.load_images_4d(files[:n], h_f, w_f, le)
        print('Data shape: {}'.format(ds.shape))

    if 1: # TRAIN ITER - DATA
        print('Training on data with iters')
        #ds = DT.subdata(ds, h, w)
        AUTO.train_data_iter(auto_network, ds, h, w, n_train=100, alpha=1e-3,
                             n_plot=4, plot_r=True, plot_i=True,
                             do_subdata=True, kmax_img=5, kmax_cost=2)

    if 0: # TRAIN NEEEEEW
        print('Training NEEEW')
        AUTO.train_new(auto_network, ds, h, w, n_train=30000, alpha=1e-3,
                             n_plot=4, plot_r=True, plot_i=True,
                             do_subdata=True, kmax_img=100, kmax_cost=20)
    
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
        AUTO.train_path_full(auto_network, files, h, w, h_f, w_f, alpha=1e-3, 
                             n_plot=n//2, do_subdata=True, kmax_cost=20)

    if 0: # LEARN
        print('Learning...?')
        #ds = np.reshape(DT.load_datas(paths.get_game_images()[:16]), (-1, h2, w2, 3))
        #print('Data shape: {}'.format(ds.shape))
        #AUTO.learn_auto_data(auto_network, ds, h, w, kmax_cost=5, slope_min=1e-3,
        #                     slope_count_max=10)

    if 0: # TEST
        n_test = 5
        print('Testing auto network for {} random images'.format(n_test))
        for i in range(n_test):
            fp = DT.get_subset(fs, n, True)
            dss = DT.subdata(FT.load_images_4d(fp, h_f, w_f, le), h, w)
            AUTO.check_auto(auto_network, dss, h, w, n // 2, True, True, i, 1)



    """ CLASS """

    if 1: # PARAMS
        class_path = paths.network_path
        size = 256
        n_classes = 8

    if 0: # LOAD
        class_network = CLASS.class_auto(name, json_data)

    if 0: # CREATE
        name = 'test'
        hidden = [64, 64, 64]
        json_data = CLASS.new(paths, name, size, hidden, n_classes)

    if 0: # TRAIN
        filepaths = paths.get_game_images()
        #ds, ls = DT.load_data_labels(paths.get_game_images(), h_d, w_d, 3,
        #                             randomize=False)
        label_names = [os.path.basename(f).split('.')[0] for f in filepaths]
        label_names = [ln.split('_')[2] for ln in label_names]
        labels = DT.to_one_hot(label_names)
        #print('Data: {}'.format(ds.shape))
        print('Labels: {}'.format(labels.shape))
        class_network = CLASS.load(name, json_data)
        CLASS.train_by_paths(class_network, auto_network, filepaths, labels,
                             h, w, h_d, w_d, n_plot=16, n_train=1000)







