import os
import numpy as np
from random import shuffle

import paths

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import DataThings as DT
from Library.General import FileThings as FT
from Library.NeuralNetworks import NetworkThings as NETS
from Library.NeuralNetworks.Autoencoder import AutoencoderAPI as AUTO
from Library.NeuralNetworks.Autoencoder import TrainAutoencoder as AUTO_T
#from Library.NeuralNetworks.Classifier import ClassifierAPI as CLASS


### PROGRAM ###

if __name__ == '__main__':


    """ GAME """

    name = 'test'
    paths.set_base(name)

    auto_path = paths.network_path
    auto_hidden = [32, 32, 32, 32, 32, 32, 64, 64]
    
    n = 64
    h = 512
    w = 512
    le = 3
    h_f = 544
    w_f = 544


    """ AUTO """

    if 0: # LOAD
        auto_network = AUTO.load_auto(name, paths.load_json())

    if 1: # CREATE
        AUTO.new(paths, name, h, w, auto_hidden, length=3, with_binary=True,
                 reuse_weights=True)

    if 1: # TRAIN - DATA
        ds = FT.load_images_4d(paths.get_game_images()[:n], h_f, w_f, le)
        print('Data shape: {}'.format(ds.shape))
        auto_network = AUTO.load(name, paths.load_json())
        AUTO.train_data(auto_network, ds, h, w, n_train=200, alpha=0.001,
                        kmax_img=20, kmax_cost=10)

    if 0: # TRAIN - PATHS
        data_path = paths.get_game_images()
        print('Number files: {}'.format(len(data_path)))
        auto_network = AUTO.load_auto(name, json_data)
        AUTO.train_auto_paths(auto_network, data_path,
                              h, w, h_d, w_d, n_train=500, alpha=0.001, n_plot=n,
                              kmax_img=50, kmax_cost=10)

    if 0: # LEARN
        ds = np.reshape(DT.load_datas(paths.get_game_images()[:16]), (-1, h2, w2, 3))
        print('Data shape: {}'.format(ds.shape))
        AUTO.learn_auto_data(auto_network, ds, h, w, kmax_cost=5, slope_min=1e-3,
                             slope_count_max=10)

    if 0: # TEST
        for i in range(10):
            fp = NETS.get_subset(paths.get_game_images(), 64, True)
            ds = FT.load_images_4d(fp, h_d, w_d, le)
            ds = DT.subdata(ds, h, w)
            AUTO_T.check_auto(auto_network, ds, h, w, 16, i, 1)


    """ CLASS - AUTO """

    if 1: # PARAMS
        class_path = paths.network_path
        size = 256
        n_classes = 8

    if 0: # LOAD
        class_network = CLASS.class_auto(name, json_data)

    if 0: # CREATE
        name = 'test'
        hidden = [64, 64, 64]
        json_data = CLASS.new_class(paths, name, size, hidden, n_classes)

    if 0: # TRAIN
        filepaths = paths.get_game_images()
        #ds, ls = DT.load_data_labels(paths.get_game_images(), h_d, w_d, 3,
        #                             randomize=False)
        label_names = [os.path.basename(f).split('.')[0] for f in filepaths]
        label_names = [ln.split('_')[2] for ln in label_names]
        labels = DT.to_one_hot(label_names)
        #print('Data: {}'.format(ds.shape))
        print('Labels: {}'.format(labels.shape))
        class_network = CLASS.load_class(name, json_data)
        CLASS.train_by_paths(class_network, auto_network, filepaths, labels,
                             h, w, h_d, w_d, n_plot=16, n_train=1000)







